# -*- coding: utf-8 -*-
"""
Merged RoboMaster DFS Maze Explorer + Marker Tracking (Digits Only)
- คงพฤติกรรมหลักของโค้ดแรก (scan-and-align ครั้งเดียว/เซลล์, L-F-R, DFS)
- เพิ่มการตรวจจับ marker เฉพาะตัวเลขในกรอบสี่เหลี่ยมแดง
- ถ้าเจอ marker หรือก้อนแดงให้หมุนกิมบอลเข้ากลาง (เฉพาะตอนมีเบาะแส)
- ผูก "เลข" ที่เจอกับกำแพงของเซลล์ แล้ววาดเลขสีแดงกลางกำแพงบนแผนที่
- รองรับตำแหน่งเฉียง ~45° ด้วยการ map x_norm -> yaw_offset -> ด้านกำแพง

หมายเหตุ:
- ต้องมี OpenCV (cv2) สำหรับจับ "ก้อนแดง" เสริมความมั่นใจตอนหา marker
- พารามิเตอร์ HFOV/เกณฑ์ต่าง ๆ ปรับได้ในส่วน Config ด้านล่าง
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from robomaster import robot, vision
import threading

try:
    import cv2
except ImportError:
    cv2 = None  # ยังรันได้ แค่จะไม่ได้ช่วยจับ "ก้อนแดง"

# ===================== Config (ปรับได้) =====================
CAMERA_HFOV_DEG = 78.0      # มุมมองกล้องแนวนอน (deg) ปรับให้ตรงรุ่นเพื่อ map มุมแม่นขึ้น
RED_MIN_AREA = 600          # px^2 ขั้นต่ำของ "ก้อนแดง" (กัน noise)
CENTER_DEADBAND = 0.06      # ภาพใกล้กลางแล้วถือว่า "พอ" ไม่หมุนเพิ่ม (~6% ของความกว้างภาพ)

# ====== DFS / Mapping Knobs ======
WALL_THRESHOLD_CM = 60
CELL_SIZE_M = 0.60
MAX_MAZE_WIDTH_M = 7 * CELL_SIZE_M

TARGET_DISTANCE_M = 0.04
TOLERANCE_M = 0.02
MAX_LATERAL_STEP_M = 0.05
MAX_FORWARD_STEP_M = 0.05

OUTSIDE_DETECT_ENABLED = True    # ใช้เพื่อวาดเส้นประนอกแผนที่
OUTSIDE_AUTO_BACKTRACK = True    # ถ้าออกนอกแผนที่ให้ backtrack ทันที


# ===================== PID Controller =====================
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self._prev_error, self._integral = 0.0, 0.0
        self._last_time = time.time()

    def compute(self, current_value):
        t = time.time()
        dt = t - self._last_time
        if dt <= 0:
            return 0.0
        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        out = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        self._prev_error, self._last_time = error, t
        return out


# ===================== Control (Robot I/O + Vision) =====================
class Control:
    def __init__(self, conn_type="ap"):
        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type=conn_type)
        self.ep_chassis = self.ep_robot.chassis
        self.ep_gimbal  = self.ep_robot.gimbal
        self.ep_sensor  = self.ep_robot.sensor
        self.ep_camera  = self.ep_robot.camera
        self.ep_vision  = self.ep_robot.vision

        # ----- State -----
        self.last_distance_cm = None
        self.current_x, self.current_y, self.current_yaw = 0.0, 0.0, 0.0

        # ----- Callbacks -----
        def _dist_cb(sub_info):
            try:
                mm = int(sub_info[0])
                if mm > 0:
                    self.last_distance_cm = mm / 10.0  # cm
            except Exception:
                pass

        def _att_cb(attitude_info):
            # yaw in degrees
            self.current_yaw = float(attitude_info[0])

        def _pos_cb(position_info):
            self.current_x, self.current_y = float(position_info[0]), float(position_info[1])

        self.ep_gimbal.recenter().wait_for_completed()
        self.ep_chassis.sub_attitude(freq=10, callback=_att_cb)
        self.ep_chassis.sub_position(freq=50, callback=_pos_cb)
        self._dist_subscribed = False
        self._dist_cb = _dist_cb

        # ---------- VISION (subscribe markers) ----------
        self._markers = []
        self._markers_lock = threading.Lock()

        def _on_markers(marker_info):
            """
            SDK มักให้ข้อมูลเป็น list ของ (x, y, w, h, info)
            - x,y,w,h: ตำแหน่ง/ขนาดในภาพ (บางเวอร์ชั่นคืนเป็น 0..1, บางทีเป็นพิกเซล)
            - info: label/ID (ของเราใช้เฉพาะ 'ตัวเลข')
            """
            now = time.time()
            parsed = []
            try:
                for item in marker_info:
                    if isinstance(item, (list, tuple)) and len(item) >= 5:
                        x, y, w, h, info = item[0], item[1], item[2], item[3], item[4]
                        parsed.append({"x": float(x), "y": float(y), "w": float(w), "h": float(h), "info": info, "ts": now})
            except Exception:
                parsed = []
            with self._markers_lock:
                self._markers = parsed

        # เปิดวิดีโอและ subscribe
        self.ep_camera.start_video_stream(display=False)
        try:
            self.ep_vision.sub_detect_info(name="marker", callback=_on_markers)
        except Exception:
            # บาง SDK ใช้ key อื่น เช่น "marker_recognize" หรือ "apriltag" (แล้วแต่ model/โมดูล)
            # ถ้าใช้ชื่ออื่นให้แก้ตรงนี้
            pass

        time.sleep(1.0)

    # ----- Marker getters -----
    def get_markers(self, max_age=0.6):
        now = time.time()
        with self._markers_lock:
            return [m for m in self._markers if now - m.get("ts", 0) <= max_age]

    # ----- Chassis/Gimbal helpers -----
    def get_yaw_deg(self):
        return self.current_yaw

    def get_xy_m(self):
        return self.current_x, self.current_y

    def _sub_distance(self, freq=20):
        if not self._dist_subscribed:
            self.ep_sensor.sub_distance(freq=freq, callback=self._dist_cb)
            self._dist_subscribed = True

    def _unsub_distance(self):
        if self._dist_subscribed:
            try:
                self.ep_sensor.unsub_distance()
            except Exception:
                pass
            self._dist_subscribed = False

    def _read_frame(self, timeout=0.25):
        # รองรับหลายเวอร์ชั่นของ SDK
        try:
            frame = self.ep_camera.read_cv2_image(strategy="newest", timeout=timeout)
            return frame
        except Exception:
            pass
        try:
            frame = self.ep_camera.read_cv2_image(timeout=timeout)
            return frame
        except Exception:
            return None

    def detect_red_blob(self):
        """หา 'ก้อนแดง' ในภาพเพื่อช่วยเล็งกิมบอลตอนยังไม่มั่นใจว่าเป็น marker"""
        if cv2 is None:
            return None
        frame = self._read_frame()
        if frame is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 100, 80]);    upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 80]);  upper2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < RED_MIN_AREA:
            return None

        x,y,w,h = cv2.boundingRect(c)
        H, W = frame.shape[:2]
        cx = x + w/2.0
        cy = y + h/2.0
        x_norm = (cx / W) * 2.0 - 1.0  # -1..1, ซ้าย=-1 ขวา=+1
        y_norm = (cy / H) * 2.0 - 1.0
        return {"x_norm": float(x_norm), "y_norm": float(y_norm), "area": float(area), "bbox": (x,y,w,h)}

    def _gimbal_center_on_x(self, x_norm, deadband=CENTER_DEADBAND):
        """
        หมุนกิมบอลเข้ากลางภาพเฉพาะแกน yaw
        x_norm: -1..1 (ลบ=ซ้าย, บวก=ขวา)
        """
        if x_norm is None or abs(x_norm) < deadband:
            self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
            return True
        yaw_rate = float(np.clip(x_norm * 150.0, -180.0, 180.0))  # deg/s
        self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=yaw_rate)
        time.sleep(0.15)
        self.ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        return False

    def try_track_marker_or_red(self, max_steps=6):
        """
        กลยุทธ์ "เห็น = หมุน":
        1) ถ้ามี marker ใน buffer -> เลือกอันใหญ่สุด -> หมุนเข้ากลาง
        2) ถ้าไม่มี marker -> ลองหา 'ก้อนแดง' -> หมุนเข้ากลาง -> รอดู marker โผล่
        ทำทีละจิ๊บ ไม่ปาดหาไร้เป้าหมาย
        """
        step = 0
        while step < max_steps:
            step += 1
            # 1) markers จาก SDK ก่อน
            mk = self.get_markers(max_age=0.6)
            if mk:
                mk = sorted(mk, key=lambda m: m["w"]*m["h"], reverse=True)[0]
                x_norm = float(mk["x"])
                # normalize ถ้า x เป็น 0..1
                if 0.0 <= x_norm <= 1.0:
                    x_norm = x_norm*2.0 - 1.0
                centered = self._gimbal_center_on_x(x_norm)
                if centered:
                    return True
                continue

            # 2) ไม่มี marker -> ใช้ก้อนแดง
            rb = self.detect_red_blob()
            if rb is None:
                break
            centered = self._gimbal_center_on_x(rb["x_norm"])
            if centered:
                time.sleep(0.12)  # รอ vision ตีความ
                if self.get_markers(max_age=0.6):
                    return True
        return False

    # ----- Distance scanning -----
    def read_distance_at(self, yaw_angle_deg, samples=5, timeout_s=1.0):
        self.last_distance_cm = None
        # หมายเหตุ: ถ้า SDK ไม่รองรับ yaw_speed ให้ลบออก
        self.ep_gimbal.moveto(pitch=0, yaw=yaw_angle_deg, yaw_speed=180).wait_for_completed()
        distances = []
        self._sub_distance()
        t0 = time.time()
        while len(distances) < samples and (time.time() - t0) < timeout_s:
            if self.last_distance_cm is not None:
                distances.append(self.last_distance_cm)
                self.last_distance_cm = None
            time.sleep(0.05)
        self._unsub_distance()
        if not distances:
            return None
        med = float(np.median(distances))
        print(f"Median distance at yaw {yaw_angle_deg}: {med:.1f} cm")
        return med

    def eye(self):
        print("Scanning: [L, F, R]")
        dist = {
            "L": self.read_distance_at(-90),
            "F": self.read_distance_at(0),
            "R": self.read_distance_at(90),
        }
        self.ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=180).wait_for_completed()
        return dist

    # ----- Motion primitives -----
    def measure_left(self):  return self.read_distance_at(-90)
    def measure_front(self): return self.read_distance_at(0)
    def measure_right(self): return self.read_distance_at(90)

    def slide_left(self, distance_m):
        print(f"Action: Sliding left {distance_m:.2f} m")
        self.ep_chassis.move(x=0, y=-distance_m, z=0).wait_for_completed()

    def slide_right(self, distance_m):
        print(f"Action: Sliding right {distance_m:.2f} m")
        self.ep_chassis.move(x=0, y= distance_m, z=0).wait_for_completed()

    def move_forward(self, distance_m):
        print(f"Action: Adjusting forward {distance_m:.2f} m")
        self.ep_chassis.move(x= distance_m, y=0, z=0).wait_for_completed()

    def move_backward(self, distance_m):
        print(f"Action: Adjusting backward {distance_m:.2f} m")
        self.ep_chassis.move(x=-distance_m, y=0, z=0).wait_for_completed()

    def stop(self):
        self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
        time.sleep(0.2)

    def turn(self, angle_deg):
        print(f"Action: Turning {angle_deg:.1f} degrees")
        self.ep_chassis.move(x=0, y=0, z=-angle_deg, z_speed=45).wait_for_completed()
        time.sleep(0.5)

    def move_forward_pid(self, cell_size_m, Kp=1.2, Ki=0.05, Kd=0.1, v_clip=0.7, tol_m=0.001):
        print(f"Action: Moving forward {cell_size_m} m")
        pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=cell_size_m)
        sx, sy = self.get_xy_m()
        while True:
            dist = math.hypot(self.current_x - sx, self.current_y - sy)
            speed = float(np.clip(pid.compute(dist), -v_clip, v_clip))
            self.ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=0.1)
            if abs(cell_size_m - dist) < tol_m:
                break
            time.sleep(0.02)
        self.stop()
        print("Movement complete.")

    def close(self):
        # cleanup
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


# ===================== Plotting =====================
plt.ion()
_fig, _ax = plt.subplots(figsize=(8, 8))

def plot_maze(current_cell, visited, walls, path_stack, title="Real-time Maze Exploration", wall_markers=None):
    ax = _ax
    ax.clear()

    # visited cells
    for (x, y) in visited:
        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, facecolor='lightgray', edgecolor='gray'))

    # walls
    if isinstance(walls, dict):
        for wall, style in walls.items():
            (x1, y1), (x2, y2) = wall
            line_style = '--' if style == 'dashed' else '-'
            color = 'r' if style == 'dashed' else 'k'
            if x1 == x2:  # horizontal line (ระหว่าง y)
                ax.plot([x1-0.5, x1+0.5], [max(y1, y2)-0.5, max(y1, y2)-0.5], color=color, linestyle=line_style, lw=4)
            else:         # vertical line (ระหว่าง x)
                ax.plot([max(x1, x2)-0.5, max(x1, x2)-0.5], [y1-0.5, y1+0.5], color=color, linestyle=line_style, lw=4)
    else:
        for wall in walls:
            (x1, y1), (x2, y2) = wall
            if x1 == x2:
                ax.plot([x1-0.5, x1+0.5], [max(y1, y2)-0.5, max(y1, y2)-0.5], 'k-', lw=4)
            else:
                ax.plot([max(x1, x2)-0.5, max(x1, x2)-0.5], [y1-0.5, y1+0.5], 'k-', lw=4)

    # <<<<<<<<<<<<<<<<<<<< START: CODE MODIFIED >>>>>>>>>>>>>>>>>>>>
    # NEW: วาด Marker ตัวเลขสีแดงบนกำแพงโดยตรง
    if wall_markers:
        for wall, label in wall_markers.items():
            (a, b) = wall
            (x1, y1), (x2, y2) = a, b
            # คำนวณหาจุดกึ่งกลางของกำแพง
            if x1 == x2:  # กำแพงแนวนอน (บน/ล่างเซลล์)
                mx = x1
                my = max(y1, y2) - 0.5
            else:         # กำแพงแนวตั้ง (ซ้าย/ขวาเซลล์)
                mx = max(x1, x2) - 0.5
                my = y1
            
            # วาดตัวเลขพร้อมพื้นหลัง (bbox) เพื่อให้มองเห็นชัดเจนและเป็นส่วนหนึ่งของกำแพง
            ax.text(mx, my, str(label), color='red', fontsize=12,
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='red', boxstyle='square,pad=0.2'))
    # <<<<<<<<<<<<<<<<<<<< END: CODE MODIFIED >>>>>>>>>>>>>>>>>>>>

    # path
    if len(path_stack) > 1:
        path_x, path_y = zip(*path_stack)
        ax.plot(path_x, path_y, 'b-o', markersize=5)

    # robot
    cx, cy = current_cell
    ax.plot(cx, cy, 'ro', markersize=12, label='Robot')

    # frame
    all_x = [c[0] for c in visited] or [0]
    all_y = [c[1] for c in visited] or [0]
    ax.set_xlim(min(all_x)-1.5, max(all_x)+1.5)
    ax.set_ylim(min(all_y)-1.5, max(all_y)+1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title(title)
    plt.pause(0.1)

def finalize_show():
    plt.ioff()
    plt.show()


# ===================== MazeSolver (DFS) =====================
class MazeSolver:
    def __init__(self, ctrl: Control):
        self.ctrl = ctrl
        self.maze_map = {}
        self.visited = set([(0, 0)])
        self.path_stack = [(0, 0)]
        self.walls = {}          # dict: {((x1,y1),(x2,y2)) : 'solid'|'dashed'}
        self.wall_markers = {}   # dict: {((x1,y1),(x2,y2)) : 'digit'}
        self.current_orientation = self._get_discretized_orientation(self.ctrl.get_yaw_deg())

    # ---- Orientation helpers ----
    @staticmethod
    def _get_discretized_orientation(yaw_deg):
        yaw = (yaw_deg + 360) % 360
        if yaw >= 315 or yaw < 45:
            return 0
        elif 45 <= yaw < 135:
            return 3
        elif 135 <= yaw < 225:
            return 2
        else:
            return 1

    @staticmethod
    def _get_target_coordinates(grid_x, grid_y, direction):
        if direction == 0:   return (grid_x,     grid_y + 1)
        elif direction == 1: return (grid_x + 1, grid_y)
        elif direction == 2: return (grid_x,     grid_y - 1)
        elif direction == 3: return (grid_x - 1, grid_y)

    @staticmethod
    def _get_relative_directions(orientation):
        return {"L": (orientation - 1 + 4) % 4, "F": orientation, "R": (orientation + 1) % 4}

    @staticmethod
    def _get_direction_to_neighbor(current_cell, target_cell):
        dx = target_cell[0] - current_cell[0]
        dy = target_cell[1] - current_cell[1]
        if dx == 1:  return 1
        if dx == -1: return 3
        if dy == 1:  return 0
        if dy == -1: return 2
        return None

    # ---- 45° side mapping ----
    @staticmethod
    def _side_from_yaw_deg(yaw_deg):
        """
        แบ่งโซนเป็น L/F/R/B จากมุมกิมบอลเทียบแกนหน้าของหุ่น (ลบ=ซ้าย, บวก=ขวา)
        [-22.5,22.5]=F ; (22.5,67.5]=R ; [-67.5,-22.5)=L ; (67.5,112.5]=R ; [-112.5,-67.5]=L ; อื่นๆ=B
        """
        a = (yaw_deg + 180) % 360 - 180
        if -22.5 <= a <= 22.5:   return "F"
        if 22.5 < a <= 67.5:     return "R"
        if -67.5 <= a < -22.5:   return "L"
        if 67.5 < a <= 112.5:    return "R"
        if -112.5 <= a < -67.5:  return "L"
        return "B"

    def _neighbor_cell_for_side(self, cell, side):
        o = self.current_orientation
        rel = {"F": o, "R": (o+1)%4, "B": (o+2)%4, "L": (o+3)%4}
        d = rel.get(side, o)
        return self._get_target_coordinates(cell[0], cell[1], d), d

    @staticmethod
    def _wall_tuple(a, b):
        return tuple(sorted((a, b)))

    # ---- Scan & Align (คงพฤติกรรมเวอร์ชันแรก) ----
    def scan_and_align(self):
        d = self.ctrl.eye()  # {"L":cm, "F":cm, "R":cm}
        l_cm, f_cm, r_cm = d.get("L"), d.get("F"), d.get("R")
        print(f"Scan distances (cm): L={l_cm}, F={f_cm}, R={r_cm}")

        # ตรวจ "นอกแผนที่" เพื่อช่วยการวาด/การตัดสินใจ backtrack
        is_outside = False
        if OUTSIDE_DETECT_ENABLED and (l_cm is not None and r_cm is not None):
            current_width_m = (l_cm + r_cm) / 100.0
            if current_width_m >= MAX_MAZE_WIDTH_M:
                is_outside = True

        l = None if l_cm is None else l_cm / 100.0
        f = None if f_cm is None else f_cm / 100.0
        r = None if r_cm is None else r_cm / 100.0

        moves = {"slide_left": 0.0, "slide_right": 0.0, "forward": 0.0, "backward": 0.0}

        # ให้หุ่นไปกลางทางถ้าเห็นซ้าย/ขวา
        if l is not None and r is not None:
            lateral_err = l - r
            if abs(lateral_err) > TOLERANCE_M:
                step = float(np.clip(abs(lateral_err) / 2.0, 0.0, MAX_LATERAL_STEP_M))
                if lateral_err > 0:
                    moves["slide_left"] = step
                else:
                    moves["slide_right"] = step
        else:
            # เกาะระยะฝั่งเดียว
            if l is not None:
                e = TARGET_DISTANCE_M - l
                if abs(e) > TOLERANCE_M:
                    if e > 0:
                        moves["slide_right"] = float(np.clip(e, 0.0, MAX_LATERAL_STEP_M))
                    else:
                        moves["slide_left"]  = float(np.clip(-e, 0.0, MAX_LATERAL_STEP_M))
            if r is not None:
                e = TARGET_DISTANCE_M - r
                if abs(e) > TOLERANCE_M:
                    if e > 0:
                        moves["slide_left"]  = max(moves["slide_left"],  float(np.clip(e, 0.0, MAX_LATERAL_STEP_M)))
                    else:
                        moves["slide_right"] = max(moves["slide_right"], float(np.clip(-e, 0.0, MAX_LATERAL_STEP_M)))

        # ด้านหน้าให้ ~TARGET_DISTANCE_M
        if f is not None:
            e = TARGET_DISTANCE_M - f
            if abs(e) > TOLERANCE_M:
                if e > 0:
                    moves["backward"] = float(np.clip(e, 0.0, MAX_FORWARD_STEP_M))
                else:
                    moves["forward"]  = float(np.clip(-e, 0.0, MAX_FORWARD_STEP_M))

        # Execute
        if moves["slide_left"] > 0:
            self.ctrl.slide_left(moves["slide_left"])
        elif moves["slide_right"] > 0:
            self.ctrl.slide_right(moves["slide_right"])

        if moves["backward"] > 0:
            self.ctrl.move_backward(moves["backward"])
        elif moves["forward"] > 0:
            self.ctrl.move_forward(moves["forward"])

        # recenter
        self.ctrl.ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=180).wait_for_completed()

        print(f"Alignment moves (m): {moves}")
        return d, is_outside

    # ---- Mapping ----
    def _scan_and_map(self, cell, distances=None, is_outside=False):
        if distances is None:
            distances, is_outside = self.scan_and_align()

        relative_dirs = self._get_relative_directions(self.current_orientation)
        open_directions = set()

        if OUTSIDE_DETECT_ENABLED and is_outside:
            print("[Mapping] Outside-of-map detected → hard fence this cell (no openings).")
            for move_key in ["L", "F", "R"]:
                direction = relative_dirs[move_key]
                dist_cm = distances.get(move_key)
                neighbor = self._get_target_coordinates(cell[0], cell[1], direction)
                wall_tuple = self._wall_tuple(cell, neighbor)
                if dist_cm is not None and dist_cm > WALL_THRESHOLD_CM:
                    self.walls[wall_tuple] = 'dashed'  # เปิดสู่ภายนอก
                else:
                    self.walls[wall_tuple] = 'solid'
            self.maze_map[cell] = open_directions
            print(f"Mapped {cell} with open directions: [] (HARD FENCE)")
            print(f"Walls updated: {len(self.walls)} total walls.")
            return

        # ปกติ
        for move_key in ["L", "F", "R"]:
            direction = relative_dirs[move_key]
            dist_cm = distances.get(move_key)
            neighbor = self._get_target_coordinates(cell[0], cell[1], direction)
            wall_tuple = self._wall_tuple(cell, neighbor)

            is_open_by_dist = dist_cm is not None and dist_cm > WALL_THRESHOLD_CM
            if is_open_by_dist:
                open_directions.add(direction)
            else:
                self.walls[wall_tuple] = 'solid'

        self.maze_map[cell] = open_directions
        print(f"Mapped {cell} with open directions: {sorted(list(open_directions))}")
        print(f"Walls updated: {len(self.walls)} total walls.")

    # ---- Marker logging (digits only) ----
    def _check_and_log_markers(self, current_cell):
        # 1) ถ้ามีเบาะแส → หมุนเข้าหาเป้าหมาย
        self.ctrl.try_track_marker_or_red(max_steps=6)

        # 2) อ่าน markers อีกรอบ
        marks = self.ctrl.get_markers(max_age=0.6)
        if not marks:
            return

        # 3) เอาเฉพาะ "เลข" ตัวเดียว (ไม่นับอย่างอื่น)
        digit_marks = []
        for m in marks:
            label = str(m.get("info", "")).strip()
            if label.isdigit() and len(label) == 1:
                digit_marks.append(m)
        if not digit_marks:
            return

        # 4) เลือกชิ้นใหญ่สุด (วางใจสุด)
        best = sorted(digit_marks, key=lambda z: z["w"]*z["h"], reverse=True)[0]

        # 5) คำนวณมุมกิมบอลจากตำแหน่ง X ในภาพ -> map เป็นด้านกำแพง
        x_norm = float(best["x"])
        if 0.0 <= x_norm <= 1.0:  # normalize 0..1 -> -1..1
            x_norm = x_norm*2.0 - 1.0
        yaw_offset = x_norm * (CAMERA_HFOV_DEG / 2.0)  # deg

        side = self._side_from_yaw_deg(yaw_offset)
        neighbor, _ = self._neighbor_cell_for_side(current_cell, side)
        wall = self._wall_tuple(current_cell, neighbor)

        # 6) บันทึกเลขลงกำแพง
        self.wall_markers[wall] = str(best["info"])
        print(f"[Map] Marker digit '{best['info']}' on side '{side}', wall={wall}")

    # ---- DFS steps ----
    def _find_and_move_to_next_cell(self, cell):
        relative_dirs = self._get_relative_directions(self.current_orientation)
        order = [relative_dirs["L"], relative_dirs["F"], relative_dirs["R"]]  # L -> F -> R
        for direction in order:
            if direction in self.maze_map.get(cell, set()):
                target_cell = self._get_target_coordinates(cell[0], cell[1], direction)
                if target_cell not in self.visited:
                    print(f"Found unvisited neighbor {target_cell}. Moving...")
                    self._turn_to(direction)
                    self.ctrl.move_forward_pid(CELL_SIZE_M)
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
        back_dir = self._get_direction_to_neighbor(current_cell, previous_cell)
        if back_dir is None:
            print("Error: Could not determine backtrack direction.")
            return False

        print(f"Backtracking from {current_cell} to {previous_cell}")
        self._turn_to(back_dir)
        self.ctrl.move_forward_pid(CELL_SIZE_M)
        return True

    def _turn_to(self, target_direction):
        turn_angle = (target_direction - self.current_orientation) * 90
        if turn_angle > 180:  turn_angle -= 360
        if turn_angle < -180: turn_angle += 360
        if abs(turn_angle) > 1:
            self.ctrl.stop()
            self.ctrl.turn(turn_angle)
            self.current_orientation = target_direction

    # ---- Main loop ----
    def explore(self):
        print("Starting DFS Maze Solver + Digit Markers...")
        while self.path_stack:
            current_cell = self.path_stack[-1]

            # 1) scan & align หนึ่งครั้ง/เซลล์
            distances, is_outside = self.scan_and_align()

            # 2) วาด map (อัปเดต real-time)
            plot_maze(current_cell, self.visited, self.walls, self.path_stack,
                      wall_markers=self.wall_markers)
            print(f"\nPosition: {current_cell}, Orientation: {self.current_orientation} (Yaw: {self.ctrl.get_yaw_deg():.1f}°)")

            # 3) log markers (เฉพาะเลขเท่านั้น) + ปักลงกำแพง
            self._check_and_log_markers(current_cell)

            # 4) บันทึกผังทางเปิด/ปิดของเซลล์นี้
            if current_cell not in self.maze_map:
                self._scan_and_map(current_cell, distances, is_outside)

            # 5) ถ้าเปิดโหมด backtrack เมื่อนอกแผนที่
            if OUTSIDE_AUTO_BACKTRACK and is_outside:
                print("[Action] Out of bounds detected. Backtracking to re-enter map.")
                if not self._backtrack():
                    break
                continue

            # 6) หาเพื่อนบ้านที่ยังไม่เคยไปตามลำดับ L-F-R
            if self._find_and_move_to_next_cell(current_cell):
                continue

            # 7) ถ้าตัน -> ถอยกลับ
            if not self._backtrack():
                break

        print("\nDFS exploration complete.")
        plot_maze(self.path_stack[-1], self.visited, self.walls, self.path_stack,
                  "Final Map", wall_markers=self.wall_markers)
        finalize_show()


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