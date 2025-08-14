import time
import math
import numpy as np
import matplotlib.pyplot as plt
from robomaster import robot, vision  # เพิ่ม vision สำหรับ marker
import threading  # ใช้ล็อก markers ให้ปลอดภัย
import cv2  # ใช้ตรวจ blob สีแดงจากภาพสด

# ===================== PID Controller Class =====================
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, i_limit=0.3, d_alpha=0.2):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self._prev_error, self._integral = 0.0, 0.0
        self._last_time = time.perf_counter()     # << ใช้ perf_counter() แทน time()
        self._d_prev = 0.0                        # สำหรับ derivative low-pass
        self.i_limit = float(i_limit)             # anti-windup (จำกัดผลรวมอินทิกรัล)
        self.d_alpha = float(d_alpha)             # 0..1, 0.2 = กรองแรง

    def compute(self, current_value):
        t = time.perf_counter()
        dt = t - self._last_time
        if dt <= 0.0: 
            return 0.0
        # ป้องกัน dt หลุด (สลับ task/GC) ไม่ให้อนุพันธ์ระเบิด
        if dt > 0.2:
            dt = 0.2

        error = self.setpoint - current_value

        # อินทิกรัล + anti-windup
        self._integral += error * dt
        if self.Ki > 0:
            i_cap = self.i_limit / self.Ki
            self._integral = max(-i_cap, min(i_cap, self._integral))

        # อนุพันธ์ + low-pass (ลด noise)
        d_raw = (error - self._prev_error) / dt
        d_filt = self.d_alpha * d_raw + (1.0 - self.d_alpha) * self._d_prev

        out = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * d_filt)

        self._prev_error = error
        self._d_prev = d_filt
        self._last_time = t
        return out

# ===================== Control Class  =====================
class Control:
    def __init__(self, conn_type="ap"):
        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type=conn_type)
        self.ep_chassis = self.ep_robot.chassis
        self.ep_gimbal  = self.ep_robot.gimbal
        self.ep_sensor  = self.ep_robot.sensor

        # ----- State -----
        self.last_distance_cm = None
        self.current_x, self.current_y, self.current_yaw = 0.0, 0.0, 0.0

        # ----- Callbacks (odometry/attitude/distance) -----
        def _dist_cb(sub_info):
            try:
                mm = int(sub_info[0])
                if mm > 0: self.last_distance_cm = mm / 10.0
            except: pass
        def _att_cb(attitude_info):
            self.current_yaw = float(attitude_info[0])
        def _pos_cb(position_info):
            self.current_x, self.current_y = float(position_info[0]), float(position_info[1])

        self.ep_gimbal.recenter().wait_for_completed()
        self.PITCH_OFFSET_DEG = -12  # Gimbal ก้มลง 
        self.ep_gimbal.moveto(pitch=self.PITCH_OFFSET_DEG, yaw=0, yaw_speed=300).wait_for_completed()  # UP SPEED
        self.ep_chassis.sub_attitude(freq=50, callback=_att_cb)
        self.ep_chassis.sub_position(freq=100, callback=_pos_cb) # เดิม 50 → 100 (ถ้าเครื่องหน่วงค่อยลดกลับ 50)
        self._dist_subscribed = False
        self._dist_cb = _dist_cb

        # ---------- VISION (Marker Detection — ไม่มี Action) ----------
        self.ep_camera = self.ep_robot.camera
        self.ep_vision = self.ep_robot.vision
        self._markers = []
        self._markers_lock = threading.Lock()

        def _on_markers(marker_info):
            now = time.time()
            with self._markers_lock:
                self._markers = [
                    {"x": x, "y": y, "w": w, "h": h, "info": info, "ts": now}
                    for (x, y, w, h, info) in marker_info
                ]

        # ต้องเปิด video stream ก่อน ถึงจะ detect ได้
        self.ep_camera.start_video_stream(display=True)
        self.ep_vision.sub_detect_info(name="marker", callback=_on_markers)

        time.sleep(1.0)

    # ----- Marker getters -----
    def get_markers(self, max_age=0.6):
        now = time.time()
        with self._markers_lock:
            return [m for m in self._markers if now - m["ts"] <= max_age]

    # ==== MARKER HELPERS ====
    def _to_px(self, x, y, frame_shape):
        """แปลงพิกัด marker ไปเป็นพิกเซลอย่างปลอดภัย (รองรับทั้ง normalized และ pixel)"""
        h, w = frame_shape[:2]
        # heuristics: ถ้าค่าอยู่ในช่วง 0..1.2 ถือว่า normalized
        if 0.0 <= float(x) <= 1.2 and 0.0 <= float(y) <= 1.2:
            cx = int(np.clip(x, 0.0, 1.0) * w)
            cy = int(np.clip(y, 0.0, 1.0) * h)
        else:
            cx = int(np.clip(x, 0, w - 1))
            cy = int(np.clip(y, 0, h - 1))
        return cx, cy

    def _filter_numeric_markers(self, markers):
        """คัดเฉพาะ marker ที่ info เป็นตัวเลขล้วน ๆ"""
        num = []
        for m in markers:
            s = str(m.get("info", "")).strip()
            if len(s) > 0 and all(ch.isdigit() for ch in s):
                num.append(m)
        return num

    def _read_frame(self):
        """อ่านเฟรมล่าสุดจากกล้อง (สำหรับตรวจ blob สีแดงแบบเร็ว ๆ)"""
        try:
            return self.ep_camera.read_cv2_image(strategy="latest")
        except:
            return None

    def _looks_red_blob(self, frame, m):
        """OPTIONAL: เช็คว่าโซนรอบ marker เป็นสีแดงเด่นจริงไหม"""
        if frame is None:
            return True
        h, w = frame.shape[:2]
        cx, cy = self._to_px(m["x"], m["y"], frame.shape)

        r  = max(6, int(0.04 * min(w, h)))
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return True
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 70, 60),   (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 70, 60), (180, 255, 255))
        red_ratio = (np.count_nonzero(mask1 | mask2) + 1) / (roi.shape[0] * roi.shape[1] + 1)
        return red_ratio > 0.12

    def _nudge_gimbal_to_x(self, x_norm, fov_deg=70.0, kp=0.9, max_step=12.0):
        """หมุนกิมบอลจึ้ก ๆ ให้ object เข้าใกล้กลางจอ"""
        err = (x_norm - 0.5) * fov_deg
        step = float(np.clip(kp * err, -max_step, max_step))
        if abs(step) < 1.0:
            return
        yaw_speed = np.sign(step) * 120
        self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=yaw_speed)
        time.sleep(min(0.12, abs(step) / 120.0))
        self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)

    def center_on_numeric_marker_once(self, markers):
        """
        ถ้าเห็น 'เลข' → เลือกอันใหญ่สุดแล้วเล็งกิมบอลเข้าใกล้
        ถ้าไม่มีเลข → (optional) เล็งเข้าหา blob สีแดงใหญ่สุดแบบหยาบ ๆ
        """
        nums = self._filter_numeric_markers(markers)
        if not nums:
            frame = self._read_frame()
            if frame is None:
                return False
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0,70,60), (10,255,255)) | cv2.inRange(hsv, (170,70,60), (180,255,255))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return False
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) < 400:
                return False
            x,y,w,h = cv2.boundingRect(c)
            x_norm = (x + w/2) / mask.shape[1]
            self._nudge_gimbal_to_x(float(x_norm))
            return True

        best = max(nums, key=lambda m: m["w"] * m["h"])
        frame = self._read_frame()
        if self._looks_red_blob(frame, best):
            self._nudge_gimbal_to_x(float(np.clip(best["x"], 0, 1)))
            return True
        return False

    # ----- Chassis/Gimbal helpers -----
    def get_yaw_deg(self): return self.current_yaw
    def get_xy_m(self): return self.current_x, self.current_y
    def _sub_distance(self, freq=20):
        if not self._dist_subscribed:
            self.ep_sensor.sub_distance(freq=freq, callback=self._dist_cb)
            self._dist_subscribed = True
    def _unsub_distance(self):
        if self._dist_subscribed:
            self.ep_sensor.unsub_distance()
            self._dist_subscribed = False
    def read_distance_at(self, yaw_angle_deg, samples=5, timeout_s=1.0):
        self.last_distance_cm = None
        # หมายเหตุ: ถ้า SDK ไม่รองรับ yaw_speed ให้ลบพารามิเตอร์นี้ออก
        self.ep_gimbal.moveto(pitch=self.PITCH_OFFSET_DEG, yaw=yaw_angle_deg, yaw_speed=300).wait_for_completed() # UP SPEED
        time.sleep(0.06)  # ให้ TOF นิ่งก่อนคัด median
        distances = []
        self._sub_distance()
        t0 = time.time()
        while len(distances) < samples and (time.time() - t0) < timeout_s:
            if self.last_distance_cm is not None:
                distances.append(self.last_distance_cm)
                self.last_distance_cm = None
            time.sleep(0.05)
        self._unsub_distance()
        if not distances: return None
        med = float(np.median(distances))
        print(f"Median distance at yaw {yaw_angle_deg}: {med:.1f} cm")
        return med
    def eye(self):
        print("Scanning: [L, F, R]")
        dist = {
            "L": self.read_distance_at(-90), "F": self.read_distance_at(0), "R": self.read_distance_at(90)
        }
        
        self.ep_gimbal.moveto(pitch=self.PITCH_OFFSET_DEG, yaw=0, yaw_speed=300).wait_for_completed() # UP SPEED
        return dist
    
    def eye_with_markers(self):
        """
        สแกนระยะ L/F/R และเก็บ marker 'แยกตามด้าน' ตอนกิมบอลชี้ด้านนั้น
        คืนค่า: (dist_cm_dict, markers_by_side)
        """
        out_dist = {}
        out_mks  = {"L": None, "F": None, "R": None}
        for yaw, key in [(-90, "L"), (0, "F"), (90, "R")]:
            d = self.read_distance_at(yaw)
            out_dist[key] = d
            cand = self.get_markers(max_age=0.5)
            cand = self._filter_numeric_markers(cand)
            if cand:
                best = max(cand, key=lambda m: m["w"] * m["h"])
                out_mks[key] = best
        self.ep_gimbal.moveto(pitch=self.PITCH_OFFSET_DEG, yaw=0, yaw_speed=300).wait_for_completed()
        return out_dist, out_mks

    
    def measure_left(self):
        return self.read_distance_at(-90)

    def measure_front(self):
        return self.read_distance_at(0)

    def measure_right(self):
        return self.read_distance_at(90)
    
    def slide_left(self, distance_m):
        print(f"Action: Sliding left {distance_m:.2f} m")
        self.ep_chassis.move(x=0, y=-distance_m, z=0, xy_speed=1.0).wait_for_completed()

    def slide_right(self, distance_m):
        print(f"Action: Sliding right {distance_m:.2f} m")
        self.ep_chassis.move(x=0, y=distance_m, z=0, xy_speed=1.0).wait_for_completed()

    def move_forward(self, distance_m):
        print(f"Action: Adjusting forward {distance_m:.2f} m")
        self.ep_chassis.move(x=distance_m, y=0, z=0, xy_speed=1.0).wait_for_completed() # UP SPEED ถ้าหุ่นลื่น/แกว่ง ลดเป็น 0.8; ถ้านิ่งมาก ค่อยดัน 1.2

    def move_backward(self, distance_m):
        print(f"Action: Adjusting backward {distance_m:.2f} m")
        self.ep_chassis.move(x=-distance_m, y=0, z=0, xy_speed=1.0).wait_for_completed()

    def stop(self):
        self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.08)
        time.sleep(0.06)

    def turn(self, angle_deg):
        print(f"Action: Turning {angle_deg:.1f} degrees")
        self.ep_chassis.move(x=0, y=0, z=-angle_deg, z_speed=120).wait_for_completed() # UP SPEED z_speed
        time.sleep(0.5)
    def move_forward_pid(self, cell_size_m, Kp=4, Ki=0.1, Kd=0.5, v_clip=0.7, tol_m=0.001, safety_stop_cm=25):
        print(f"Action: Moving forward {cell_size_m} m")
        pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=cell_size_m)
        sx, sy = self.get_xy_m()
        last_safety_check = time.time()
        while True:
            # เซฟตี้: เช็ค ToF ระหว่างทางทุก ~60ms
            if self.last_distance_cm is not None and self.last_distance_cm < safety_stop_cm:
                self.stop()
                print(f"[SAFETY] Obstacle {self.last_distance_cm:.1f} cm ahead. Aborting forward.")
                break

            dist = math.hypot(self.current_x - sx, self.current_y - sy)
            speed = float(np.clip(pid.compute(dist), -v_clip, v_clip))
            self.ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=0.1)
            if abs(cell_size_m - dist) < tol_m:
                break
            time.sleep(0.02)
        self.stop()
        print("Movement complete.")


    def close(self):
        # ปิดให้สะอาดทุกตัว
        try: self.ep_sensor.unsub_distance()
        except: pass
        try: self.ep_chassis.unsub_attitude()
        except: pass
        try: self.ep_chassis.unsub_position()
        except: pass
        try: self.ep_vision.unsub_detect_info(name="marker")
        except: pass
        try: self.ep_camera.stop_video_stream()
        except: pass
        try: self.ep_robot.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

# ===================== Plotting Functions  =====================
plt.ion()
_fig, _ax = plt.subplots(figsize=(8, 8))
def plot_maze(current_cell, visited, walls, path_stack, title="Real-time Maze Exploration", wall_markers=None):
    ax = _ax; ax.clear()
    # วาดช่องที่ไปมาแล้ว
    for (x, y) in visited:
        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, facecolor='lightgray', edgecolor='gray'))

    # วาดกำแพงจาก self.walls (ยังเป็น set() ได้)
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        if x1 == x2:
            ax.plot([x1-0.5, x1+0.5], [max(y1, y2)-0.5, max(y1, y2)-0.5], 'k-', lw=4)
        else:
            ax.plot([max(x1, x2)-0.5, max(x1, x2)-0.5], [y1-0.5, y1+0.5], 'k-', lw=4)

    # >>> วาง "เลขสีแดง" กลางกำแพง (ดึงจาก self.wall_markers)
    if wall_markers:
        for wall, text in wall_markers.items():
            (wx1, wy1), (wx2, wy2) = wall
            mx, my = (wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0
            ax.text(mx, my, str(text), color='red', fontsize=14, fontweight='bold',
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # วาดเส้นทาง+ตำแหน่งหุ่น
    if len(path_stack) > 1:
        path_x, path_y = zip(*path_stack)
        ax.plot(path_x, path_y, 'b-o', markersize=5)
    cx, cy = current_cell
    ax.plot(cx, cy, 'ro', markersize=12, label='Robot')

    # กรอบภาพ
    all_x = [c[0] for c in visited] or [0]
    all_y = [c[1] for c in visited] or [0]
    ax.set_xlim(min(all_x)-1.5, max(all_x)+1.5)
    ax.set_ylim(min(all_y)-1.5, max(all_y)+1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title(title)
    plt.pause(0.1)

def finalize_show():
    plt.ioff(); plt.show()

# ===================== MazeSolver Class =====================
class MazeSolver:
    WALL_THRESHOLD = 60
    CELL_SIZE = 0.6
    MAX_MAZE_WIDTH_M = 7 * 0.6  # เกณฑ์ความกว้างที่ถือว่าเริ่มออกนอกเขาวงกต


    # >>> NEW: ค่าคอนฟิกการจัดตำแหน่ง
    TARGET_DISTANCE_M = 0.08    # อยากห่างกำแพง ~8 ซม.
    TOLERANCE_M = 0.02           # ยอมผิดได้ ±2 ซม.
    MAX_LATERAL_STEP_M = 0.08    # จำกัดระยะสไลด์ต่อครั้ง          UP SPEED
    MAX_FORWARD_STEP_M = 0.08    # จำกัดระยะเดินหน้า/ถอยต่อครั้ง    UP SPEED

    def __init__(self, ctrl: Control):
        self.ctrl = ctrl
        self.maze_map = {}
        self.visited = set([(0, 0)])
        self.path_stack = [(0, 0)]
        self.walls = set()
        self.wall_markers = {}  # {(cellA, cellB): '7', ...}
        self.current_orientation = self._get_discretized_orientation(self.ctrl.get_yaw_deg())

    @staticmethod
    def _get_discretized_orientation(yaw_deg):
        yaw = (yaw_deg + 360) % 360
        if yaw >= 315 or yaw < 45:   return 0   # เหนือ  (y+)
        elif 45 <= yaw < 135:        return 1   # ตะวันออก (x+)
        elif 135 <= yaw < 225:       return 2   # ใต้   (y-)
        else:                        return 3   # ตะวันตก (x-)


    @staticmethod
    def _get_target_coordinates(grid_x, grid_y, direction):
        if direction == 0:   return (grid_x, grid_y + 1)
        elif direction == 1: return (grid_x + 1, grid_y)
        elif direction == 2: return (grid_x, grid_y - 1)
        elif direction == 3: return (grid_x - 1, grid_y)

    @staticmethod
    def _get_relative_directions(orientation):
        return {"L": (orientation - 1 + 4) % 4, "F": orientation, "R": (orientation + 1) % 4}

    @staticmethod
    def _get_direction_to_neighbor(current_cell, target_cell):
        dx = target_cell[0] - current_cell[0]
        dy = target_cell[1] - current_cell[1]
        if dx == 1: return 1
        if dx == -1: return 3
        if dy == 1: return 0
        if dy == -1: return 2
        return None

    # >>> NEW: สแกนครั้งเดียว + จัดตำแหน่งจากระยะเดียวกัน
    def scan_and_align(self):
        """
        สแกน L/F/R ครั้งเดียวด้วยกิมบอล + จัดตำแหน่ง
        คืนค่า: (distances_cm_dict, markers_by_side, is_outside)
        """
        distances, mks = self.ctrl.eye_with_markers()  # ได้ "ระยะ" + "marker แยกตามด้าน"
        l_cm, f_cm, r_cm = distances.get("L"), distances.get("F"), distances.get("R")
        print(f"Scan distances (cm): L={l_cm}, F={f_cm}, R={r_cm}")

        # ---- ตรวจ out-of-bounds: เปิดทั้ง L/F/R และกว้างเกิน ----
        is_outside = False
        is_l_open = l_cm is not None and l_cm > self.WALL_THRESHOLD
        is_f_open = f_cm is not None and f_cm > self.WALL_THRESHOLD
        is_r_open = r_cm is not None and r_cm > self.WALL_THRESHOLD
        if is_l_open and is_f_open and is_r_open and (l_cm is not None) and (r_cm is not None):
            current_width_m = (l_cm + r_cm) / 100.0
            if current_width_m >= self.MAX_MAZE_WIDTH_M:
                is_outside = True

        # ---- Align เหมือนเดิม ----
        l = None if l_cm is None else l_cm / 100.0
        f = None if f_cm is None else f_cm / 100.0
        r = None if r_cm is None else r_cm / 100.0
        moves = {"slide_left": 0.0, "slide_right": 0.0, "forward": 0.0, "backward": 0.0}

        if l is not None and r is not None:
            lateral_err = l - r
            if abs(lateral_err) > self.TOLERANCE_M:
                step = float(np.clip(abs(lateral_err)/2.0, 0.0, self.MAX_LATERAL_STEP_M))
                if lateral_err > 0: moves["slide_left"] = step
                else:                moves["slide_right"] = step
        else:
            if l is not None:
                e = self.TARGET_DISTANCE_M - l
                if abs(e) > self.TOLERANCE_M:
                    if e > 0:  moves["slide_right"] = float(np.clip(e, 0.0, self.MAX_LATERAL_STEP_M))
                    else:      moves["slide_left"]  = float(np.clip(-e, 0.0, self.MAX_LATERAL_STEP_M))
            if r is not None:
                e = self.TARGET_DISTANCE_M - r
                if abs(e) > self.TOLERANCE_M:
                    if e > 0:  moves["slide_left"]  = max(moves["slide_left"],  float(np.clip(e, 0.0, self.MAX_LATERAL_STEP_M)))
                    else:      moves["slide_right"] = max(moves["slide_right"], float(np.clip(-e, 0.0, self.MAX_LATERAL_STEP_M)))
        if f is not None:
            e = self.TARGET_DISTANCE_M - f
            if abs(e) > self.TOLERANCE_M:
                if e > 0:  moves["backward"] = float(np.clip(e, 0.0, self.MAX_FORWARD_STEP_M))
                else:      moves["forward"]  = float(np.clip(-e, 0.0, self.MAX_FORWARD_STEP_M))

        if moves["slide_left"] > 0:   self.ctrl.slide_left(moves["slide_left"])
        elif moves["slide_right"] > 0:self.ctrl.slide_right(moves["slide_right"])
        if moves["backward"] > 0:     self.ctrl.move_backward(moves["backward"])
        elif moves["forward"] > 0:    self.ctrl.move_forward(moves["forward"])

        # หมุนกิมบอลเข้าหา marker เลข/แดงแบบจึ้กเดียว (ไม่เจอ = ไม่หมุน)
        seen_now = self.ctrl.get_markers(max_age=0.3)
        did_center = self.ctrl.center_on_numeric_marker_once(seen_now)
        if did_center:
            time.sleep(0.1)


        print(f"Alignment moves (m): {moves}")
        return distances, mks, is_outside


    def explore(self):
        print("Starting DFS Maze Solver (marker-aware)...")
        while self.path_stack:
            current_cell = self.path_stack[-1]

            # สแกน + จัดตำแหน่ง + ได้ marker แยกด้าน + เช็ค out-of-bounds
            distances, mks_by_side, is_outside = self.scan_and_align()

            # วาดแผนที่พร้อมเลขกลางกำแพง
            plot_maze(current_cell, self.visited, self.walls, self.path_stack, wall_markers=self.wall_markers)
            print(f"\nPosition: {current_cell}, Orientation: {self.current_orientation} (Yaw: {self.ctrl.get_yaw_deg():.1f}°)")

            # log สั้น ๆ เฉย ๆ
            markers = self.ctrl.get_markers(max_age=0.6)
            if markers:
                ids = [str(m["info"]) for m in markers]
                xs = [round(m["x"], 3) for m in markers]
                print(f"[Marker] seen={len(markers)} ids={ids} x={xs}")

            # map cell นี้ด้วยระยะ + marker เดิม (ไม่หมุนซ้ำ)
            if current_cell not in self.maze_map:
                self._scan_and_map(current_cell, distances, mks_by_side, is_outside)

            # ถ้าออกนอกเขาวงกต → ถอยกลับเข้ามาก่อน
            if is_outside:
                print("[Action] Out of bounds detected → Backtrack")
                if not self._backtrack():
                    break
                continue

            # เดินต่อแบบ DFS
            if self._find_and_move_to_next_cell(current_cell):
                continue

            # หรือถ้าตันก็ backtrack
            if not self._backtrack():
                break

        print("\nDFS exploration complete.")
        plot_maze(self.path_stack[-1], self.visited, self.walls, self.path_stack, "Final Map", wall_markers=self.wall_markers)
        finalize_show()


    # >>> CHANGED: รับระยะที่สแกนมาก่อนหน้า (cm) ถ้าไม่ส่งมาจะสแกนเอง
    def _scan_and_map(self, cell, distances=None, markers_by_side=None, is_outside=False):
        if distances is None:
            print(f"Cell {cell} is unmapped. Scanning...")
            distances, markers_by_side = self.ctrl.eye_with_markers()

        relative_dirs = self._get_relative_directions(self.current_orientation)
        open_directions = set()

        for move_key in ["L", "F", "R"]:
            direction = relative_dirs[move_key]
            dist_cm = distances.get(move_key)
            neighbor = self._get_target_coordinates(cell[0], cell[1], direction)
            wall_tuple = tuple(sorted((cell, neighbor)))

            if dist_cm is not None and dist_cm > self.WALL_THRESHOLD:
                # เปิดเป็นทางเดิน
                open_directions.add(direction)
            else:
                # เป็นกำแพง → บันทึกกำแพง
                self.walls.add(wall_tuple)

                # ถ้ามี marker ฝั่งนี้และเป็น "เลข" → ปักเลขกลางกำแพง
                if markers_by_side and markers_by_side.get(move_key):
                    info = str(markers_by_side[move_key]["info"]).strip()
                    if info.isdigit():
                        self.wall_markers[wall_tuple] = info

        self.maze_map[cell] = open_directions
        print(f"Mapped {cell} with open directions: {sorted(list(open_directions))}")


    def _find_and_move_to_next_cell(self, cell):
        relative_dirs = self._get_relative_directions(self.current_orientation)
        # กลับไปใช้ลำดับเดิม L -> F -> R แบบไม่ยุ่งกับ marker
        search_order = [relative_dirs["L"], relative_dirs["F"], relative_dirs["R"]]

        for direction in search_order:
            if direction in self.maze_map.get(cell, set()):
                target_cell = self._get_target_coordinates(cell[0], cell[1], direction)
                if target_cell not in self.visited:
                    print(f"Found unvisited neighbor {target_cell}. Moving...")
                    self._turn_to(direction)
                    self.ctrl.move_forward_pid(self.CELL_SIZE)
                    self.visited.add(target_cell)
                    self.path_stack.append(target_cell)
                    return True
        return False

    def _backtrack(self):
        print("Dead end. Backtracking...")
        if len(self.path_stack) <= 1:
            print("Returned to start. Exploration finished.")
            return False

        current_cell = self.path_stack.pop()
        previous_cell = self.path_stack[-1]
        backtrack_direction = self._get_direction_to_neighbor(current_cell, previous_cell)
        if backtrack_direction is None:
            print("Error: Could not determine backtrack direction.")
            return False

        print(f"Backtracking from {current_cell} to {previous_cell}")
        self._turn_to(backtrack_direction)
        self.ctrl.move_forward_pid(self.CELL_SIZE)
        return True

    def _turn_to(self, target_direction):
        turn_angle = (target_direction - self.current_orientation) * 90
        if turn_angle > 180: turn_angle -= 360
        if turn_angle < -180: turn_angle += 360
        if abs(turn_angle) > 1:
            self.ctrl.stop()
            self.ctrl.turn(turn_angle)
            self.current_orientation = target_direction

# ===================== Main =====================
if __name__ == "__main__":
    ctrl = None
    try:
        print("Connecting to robot...")
        ctrl = Control(conn_type="ap")
        print("Robot connected. Initializing solver...")
        solver = MazeSolver(ctrl)
        solver.explore()
    except Exception as e:
        print(f"\n--- An error occurred in the main loop: {e} ---")
        import traceback
        traceback.print_exc()
    finally:
        if ctrl:
            print("Cleaning up and closing connection.")
            ctrl.close()
