import time
import math
import numpy as np
import matplotlib.pyplot as plt
from robomaster import robot, vision
import threading


# ===================== PID Controller Class =====================
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self._prev_error, self._integral = 0.0, 0.0
        self._last_time = time.time()

    def compute(self, current_value):
        t, dt = time.time(), time.time() - self._last_time
        if dt <= 0:
            return 0.0
        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt
        out = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)
        self._prev_error, self._last_time = error, t
        return out


# ===================== Control Class =====================
class Control:
    def __init__(self, conn_type="ap"):
        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type=conn_type)
        self.ep_chassis = self.ep_robot.chassis
        self.ep_gimbal = self.ep_robot.gimbal
        self.ep_sensor = self.ep_robot.sensor

        # ----- State -----
        self.last_distance_cm = None
        self.current_x, self.current_y, self.current_yaw = 0.0, 0.0, 0.0

        # ----- Callbacks (odometry/attitude/distance) -----
        def _dist_cb(sub_info):
            try:
                mm = int(sub_info[0])
                if mm > 0:
                    self.last_distance_cm = mm / 10.0
            except:
                pass

        def _att_cb(attitude_info):
            self.current_yaw = float(attitude_info[0])

        def _pos_cb(position_info):
            self.current_x, self.current_y = float(position_info[0]), float(position_info[1])

        self.ep_gimbal.recenter().wait_for_completed()
        self.ep_chassis.sub_attitude(freq=10, callback=_att_cb)
        self.ep_chassis.sub_position(freq=50, callback=_pos_cb)
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

        self.ep_camera.start_video_stream(display=False)
        self.ep_vision.sub_detect_info(name="marker", callback=_on_markers)
        time.sleep(1.0)

        # ----- Marker detection state -----
        self.marker_detect_state = None  # NEW: Track marker detection state
        self.last_marker_info = None     # NEW: Store last detected marker info

    # NEW: Add marker detection and gimbal control methods
    def detect_and_verify_marker(self, initial_angle=0):
        """Detect markers and verify using gimbal control"""
        markers = self.get_markers(max_age=0.3)
        if not markers:
            return None
        
        # Filter for numeric markers only
        numeric_markers = []
        for m in markers:
            try:
                int(m["info"])  # Try to convert to integer
                numeric_markers.append(m)
            except ValueError:
                continue
        
        if not numeric_markers:
            return None

        # Sort by size (closest marker has largest size)
        marker = max(numeric_markers, key=lambda m: m["w"] * m["h"])
        
        # Center gimbal on marker for better detection
        screen_center_x = 1280 / 2
        marker_center_x = marker["x"]
        angle_offset = (marker_center_x - screen_center_x) / screen_center_x * 30
        target_angle = initial_angle + angle_offset
        
        self.ep_gimbal.moveto(pitch=0, yaw=target_angle, yaw_speed=60).wait_for_completed()
        time.sleep(0.2)  # Allow camera to stabilize
        
        # Verify marker after centering
        verify_markers = self.get_markers(max_age=0.3)
        if not verify_markers:
            return None
            
        # Return verified marker info
        try:
            marker_num = int(marker["info"])
            return {
                "number": marker_num,
                "angle": target_angle,
                "position": (marker["x"], marker["y"]),
                "size": (marker["w"], marker["h"])
            }
        except ValueError:
            return None

    # ----- Marker getters -----
    def get_markers(self, max_age=0.6):
        now = time.time()
        with self._markers_lock:
            return [m for m in self._markers if now - m["ts"] <= max_age]

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
            self.ep_sensor.unsub_distance()
            self._dist_subscribed = False

    def read_distance_at(self, yaw_angle_deg, samples=5, timeout_s=1.0):
        self.last_distance_cm = None
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

    def slide_left(self, distance_m):
        print(f"Action: Sliding left {distance_m:.2f} m")
        self.ep_chassis.move(x=0, y=-distance_m, z=0).wait_for_completed()

    def slide_right(self, distance_m):
        print(f"Action: Sliding right {distance_m:.2f} m")
        self.ep_chassis.move(x=0, y=distance_m, z=0).wait_for_completed()

    def move_forward(self, distance_m):
        print(f"Action: Adjusting forward {distance_m:.2f} m")
        self.ep_chassis.move(x=distance_m, y=0, z=0).wait_for_completed()

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

    def move_forward_pid(self, cell_size_m, Kp=4, Ki=0.1, Kd=0.5, v_clip=0.7, tol_m=0.001):
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
        try:
            self.ep_sensor.unsub_distance()
        except: pass
        try:
            self.ep_chassis.unsub_attitude()
        except: pass
        try:
            self.ep_chassis.unsub_position()
        except: pass
        try:
            self.ep_vision.unsub_detect_info(name="marker")
        except: pass
        try:
            self.ep_camera.stop_video_stream()
        except: pass
        try:
            self.ep_robot.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")


# ===================== Plotting Functions =====================
plt.ion()
_fig, _ax = plt.subplots(figsize=(8, 8))


def plot_maze(current_cell, visited, walls, path_stack, markers={}, title="Real-time Maze Exploration"):
    ax = _ax
    ax.clear()
    
    # วาดพื้นที่ที่เคยเดินผ่าน
    for x, y in visited:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightgray', edgecolor='gray'))
    
    # วาดกำแพง
    for wall, style in walls.items():
        (x1, y1), (x2, y2) = wall
        line_style = '--' if style == 'dashed' else '-'
        color = 'r' if style == 'dashed' else 'k'
        
        if x1 == x2:  # กำแพงแนวนอน
            ax.plot([x1 - 0.5, x1 + 0.5], [max(y1, y2) - 0.5, max(y1, y2) - 0.5], 
                   color=color, linestyle=line_style, lw=2)
            
            # ถ้ามี marker ที่กำแพงนี้ ให้วาดสี่เหลี่ยมสีแดง
            if wall in markers:
                marker_rect = plt.Rectangle(
                    (x1 - 0.15, max(y1, y2) - 0.65),  # ตำแหน่งเริ่มต้น
                    0.3, 0.3,  # ขนาด
                    facecolor='red',
                    edgecolor='darkred',
                    alpha=0.7,
                    zorder=3  # แสดงทับกำแพง
                )
                ax.add_patch(marker_rect)
                # เพิ่มตัวเลข marker
                ax.text(x1, max(y1, y2) - 0.5, str(markers[wall]),
                       color='white', ha='center', va='center',
                       fontweight='bold', fontsize=10, zorder=4)
                
        else:  # กำแพงแนวตั้ง
            ax.plot([max(x1, x2) - 0.5, max(x1, x2) - 0.5], [y1 - 0.5, y1 + 0.5], 
                   color=color, linestyle=line_style, lw=2)
            
            # ถ้ามี marker ที่กำแพงนี้ ให้วาดสี่เหลี่ยมสีแดง
            if wall in markers:
                marker_rect = plt.Rectangle(
                    (max(x1, x2) - 0.65, y1 - 0.15),  # ตำแหน่งเริ่มต้น
                    0.3, 0.3,  # ขนาด
                    facecolor='red',
                    edgecolor='darkred',
                    alpha=0.7,
                    zorder=3  # แสดงทับกำแพง
                )
                ax.add_patch(marker_rect)
                # เพิ่มตัวเลข marker
                ax.text(max(x1, x2) - 0.5, y1, str(markers[wall]),
                       color='white', ha='center', va='center',
                       fontweight='bold', fontsize=10, zorder=4)

    # วาดเส้นทางและตำแหน่งหุ่นยนต์
    if len(path_stack) > 1:
        path_x, path_y = zip(*path_stack)
        ax.plot(path_x, path_y, 'b-o', markersize=5)
    
    cx, cy = current_cell
    ax.plot(cx, cy, 'ro', markersize=12, label='Robot')
    
    # ตั้งค่าขอบเขตและการแสดงผล
    all_x = [c[0] for c in visited] or [0]
    all_y = [c[1] for c in visited] or [0]
    ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
    ax.set_ylim(min(all_y) - 1.5, max(all_y) + 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title(title)
    plt.pause(0.1)


def finalize_show():
    plt.ioff()
    plt.show()


# ===================== MazeSolver Class =====================
class MazeSolver:
    WALL_THRESHOLD = 60
    CELL_SIZE = 0.6
    MAX_MAZE_WIDTH_M = 7 * 0.6

    TARGET_DISTANCE_M = 0.04
    TOLERANCE_M = 0.02
    MAX_LATERAL_STEP_M = 0.05
    MAX_FORWARD_STEP_M = 0.05

    def __init__(self, ctrl: Control):
        self.ctrl = ctrl
        self.maze_map = {}
        self.visited = set([(0, 0)])
        self.path_stack = [(0, 0)]
        self.walls = {}  # >>> CHANGED: ใช้ dict เพื่อเก็บ style ของกำแพง
        self.current_orientation = self._get_discretized_orientation(self.ctrl.get_yaw_deg())
        self.marker_positions = {}  # NEW: Store marker locations

    @staticmethod
    def _get_discretized_orientation(yaw_deg):
        yaw = (yaw_deg + 360) % 360
        if yaw >= 315 or yaw < 45: return 0
        elif 45 <= yaw < 135: return 3
        elif 135 <= yaw < 225: return 2
        else: return 1

    @staticmethod
    def _get_target_coordinates(grid_x, grid_y, direction):
        if direction == 0: return (grid_x, grid_y + 1)
        elif direction == 1: return (grid_x + 1, grid_y)
        elif direction == 2: return (grid_x, grid_y - 1)
        elif direction == 3: return (grid_x - 1, grid_y)

    @staticmethod
    def _get_relative_directions(orientation):
        return {"L": (orientation - 1 + 4) % 4, "F": orientation, "R": (orientation + 1) % 4}

    @staticmethod
    def _get_direction_to_neighbor(current_cell, target_cell):
        dx, dy = target_cell[0] - current_cell[0], target_cell[1] - current_cell[1]
        if dx == 1: return 1
        if dx == -1: return 3
        if dy == 1: return 0
        if dy == -1: return 2
        return None

    def scan_and_align(self):
        d = self.ctrl.eye()
        l_cm, f_cm, r_cm = d.get("L"), d.get("F"), d.get("R")
        print(f"Scan distances (cm): L={l_cm}, F={f_cm}, R={r_cm}")

        # >>> NEW: ตรวจสอบและคืนค่าสถานะ "นอกแผนที่"
        is_outside = False
        is_l_open = l_cm is not None and l_cm > self.WALL_THRESHOLD
        is_f_open = f_cm is not None and f_cm > self.WALL_THRESHOLD
        is_r_open = r_cm is not None and r_cm > self.WALL_THRESHOLD
        
        if is_l_open and is_f_open and is_r_open:
            current_width_m = (l_cm + r_cm) / 100.0
            if current_width_m >= self.MAX_MAZE_WIDTH_M:
                is_outside = True
        
        # --- ส่วนของการจัดตำแหน่ง (Align) ยังทำงานเหมือนเดิม ---
        l = None if l_cm is None else l_cm / 100.0
        f = None if f_cm is None else f_cm / 100.0
        r = None if r_cm is None else r_cm / 100.0
        moves = {"slide_left": 0.0, "slide_right": 0.0, "forward": 0.0, "backward": 0.0}

        if l is not None and r is not None:
            lateral_err = l - r
            if abs(lateral_err) > self.TOLERANCE_M:
                step = float(np.clip(abs(lateral_err) / 2.0, 0.0, self.MAX_LATERAL_STEP_M))
                if lateral_err > 0: moves["slide_left"] = step
                else: moves["slide_right"] = step
        # ... (ส่วนที่เหลือของการจัดตำแหน่ง) ...

        if moves["slide_left"] > 0: self.ctrl.slide_left(moves["slide_left"])
        elif moves["slide_right"] > 0: self.ctrl.slide_right(moves["slide_right"])
        if moves["backward"] > 0: self.ctrl.move_backward(moves["backward"])
        elif moves["forward"] > 0: self.ctrl.move_forward(moves["forward"])

        self.ctrl.ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=180).wait_for_completed()
        print(f"Alignment moves (m): {moves}")
        
        return d, is_outside # >>> CHANGED: คืนค่าสถานะนอกแผนที่ไปด้วย

    def explore(self):
        print("Starting DFS Maze Solver...")
        while self.path_stack:
            current_cell = self.path_stack[-1]

            # >>> CHANGED: รับสถานะ is_outside จากการสแกน
            distances, is_outside = self.scan_and_align()

            plot_maze(current_cell, self.visited, self.walls, self.path_stack, 
                      markers=self.marker_positions)
            print(f"\nPosition: {current_cell}, Orientation: {self.current_orientation} (Yaw: {self.ctrl.get_yaw_deg():.1f}°)")

            markers = self.ctrl.get_markers(max_age=0.6)
            if markers:
                ids = [str(m["info"]) for m in markers]
                print(f"[Marker] seen={len(markers)} ids={ids}")

            # ทำการ map ก่อนเสมอ เพื่อบันทึกกำแพง (อาจเป็นเส้นประ)
            if current_cell not in self.maze_map:
                self._scan_and_map(current_cell, distances, is_outside)

            # >>> NEW: ถ้าอยู่นอกแผนที่ ให้ backtrack กลับเข้ามาก่อน
            if is_outside:
                print("[Action] Out of bounds detected. Backtracking to re-enter map.")
                self._backtrack()
                continue # ข้ามไปรอบถัดไปเพื่อประเมินสถานการณ์ใหม่

            if self._find_and_move_to_next_cell(current_cell):
                continue

            if not self._backtrack():
                break

        print("\nDFS exploration complete.")
        plot_maze(self.path_stack[-1], self.visited, self.walls, self.path_stack, "Final Map")
        finalize_show()

    def _scan_and_map(self, cell, distances=None, is_outside=False):
        if distances is None: # กรณีเรียกใช้โดยไม่ผ่าน explore loop หลัก
            distances, is_outside = self.scan_and_align()

        if is_outside:
            print("[Mapping] Applying out-of-bounds rule (dashed walls for L/R).")

        relative_dirs = self._get_relative_directions(self.current_orientation)
        open_directions = set()

        for move_key in ["L", "F", "R"]:
            direction = relative_dirs[move_key]
            dist_cm = distances.get(move_key)
            neighbor = self._get_target_coordinates(cell[0], cell[1], direction)
            wall_tuple = tuple(sorted((cell, neighbor)))

            is_open_by_dist = dist_cm is not None and dist_cm > self.WALL_THRESHOLD

            # >>> CHANGED: ตรวจสอบและกำหนด style ของกำแพง
            if is_open_by_dist and is_outside and move_key in ['L', 'R']:
                # เป็นพื้นที่เปิดโล่งนอกแผนที่ -> สร้างกำแพงเส้นประ
                self.walls[wall_tuple] = 'dashed'
            elif is_open_by_dist:
                # เป็นทางเปิดปกติ
                open_directions.add(direction)
            else:
                # เป็นกำแพงทึบปกติ
                self.walls[wall_tuple] = 'solid'

        self.maze_map[cell] = open_directions
        print(f"Mapped {cell} with open directions: {sorted(list(open_directions))}")
        print(f"Walls updated: {len(self.walls)} total walls.")

        # NEW: Check for markers at each wall
        relative_dirs = self._get_relative_directions(self.current_orientation)
        for move_key in ["L", "F", "R"]:
            direction = relative_dirs[move_key]
            marker_info = self.ctrl.detect_and_verify_marker(
                initial_angle=-90 if move_key == "L" else 0 if move_key == "F" else 90
            )
            if marker_info:
                neighbor = self._get_target_coordinates(cell[0], cell[1], direction)
                wall_pos = tuple(sorted((cell, neighbor)))
                self.marker_positions[wall_pos] = marker_info["number"]
                print(f"Found marker {marker_info['number']} at wall {wall_pos}")

    def _find_and_move_to_next_cell(self, cell):
        relative_dirs = self._get_relative_directions(self.current_orientation)
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
        print("Backtracking...")
        if len(self.path_stack) <= 1:
            print("Returned to start. Exploration finished.")
            return False

        current_cell = self.path_stack.pop()
        previous_cell = self.path_stack[-1]
        backtrack_direction = self._get_direction_to_neighbor(current_cell, previous_cell)
        
        if backtrack_direction is None:
            print("Error: Could not determine backtrack direction.")
            self.path_stack.append(current_cell) # คืนค่าเดิมกลับไปใน stack
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