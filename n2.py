import time
import math
import numpy as np
import matplotlib.pyplot as plt
from robomaster import robot, vision
import threading
import cv2


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

        # ---------- VISION (Marker Detection & Red Object Detection) ----------
        self.ep_camera = self.ep_robot.camera
        self.ep_vision = self.ep_robot.vision
        self._markers = []
        self._markers_lock = threading.Lock()

        # Red object detection variables
        self._red_objects = []
        self._red_objects_lock = threading.Lock()
        self._current_frame = None
        self._frame_lock = threading.Lock()

        def _on_markers(marker_info):
            now = time.time()
            with self._markers_lock:
                self._markers = [
                    {"x": x, "y": y, "w": w, "h": h, "info": info, "ts": now}
                    for (x, y, w, h, info) in marker_info
                ]

        def _on_video_frame(frame):
            with self._frame_lock:
                self._current_frame = frame

        self.ep_camera.start_video_stream(display=False)
        self.ep_vision.sub_detect_info(name="marker", callback=_on_markers)
        # Subscribe to video stream for red object detection
        self.ep_camera.set_fps(fps="medium")
        time.sleep(1.0)

    # ----- Marker getters -----
    def get_markers(self, max_age=0.6):
        now = time.time()
        with self._markers_lock:
            # Filter only numeric markers (0-9)
            numeric_markers = []
            for m in self._markers:
                if now - m["ts"] <= max_age:
                    try:
                        # Check if marker info is numeric (0-9)
                        marker_id = str(m["info"])
                        if marker_id.isdigit() and len(marker_id) == 1:
                            numeric_markers.append(m)
                    except:
                        continue
            return numeric_markers

    def detect_red_objects(self):
        """Detect red objects in current frame using color segmentation"""
        with self._frame_lock:
            if self._current_frame is None:
                return []
        
        frame = self._current_frame.copy()
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for red color in HSV
        # Red has two ranges in HSV due to hue wrapping
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        red_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                red_objects.append({
                    "x": center_x,
                    "y": center_y, 
                    "w": w,
                    "h": h,
                    "area": area
                })
        
        return red_objects

    def scan_for_markers_and_red_objects(self):
        """Enhanced scanning that looks for markers and red objects in all directions"""
        print("=== Enhanced Marker & Red Object Scanning ===")
        found_markers = []
        
        # Scan positions: Left (-90°), Front (0°), Right (90°)
        scan_positions = [("Left", -90), ("Front", 0), ("Right", 90)]
        
        for direction_name, yaw_angle in scan_positions:
            print(f"\n--- Scanning {direction_name} (Yaw: {yaw_angle}°) ---")
            
            # Move gimbal to position
            self.ep_gimbal.moveto(pitch=0, yaw=yaw_angle, yaw_speed=180).wait_for_completed()
            time.sleep(0.5)  # Allow gimbal to settle
            
            # Update current frame
            time.sleep(0.2)
            
            # Check for standard markers first
            markers = self.get_markers(max_age=0.3)
            if markers:
                print(f"Found {len(markers)} numeric markers at {direction_name}")
                for marker in markers:
                    marker_info = {
                        "direction": direction_name,
                        "yaw": yaw_angle,
                        "id": str(marker["info"]),
                        "position": (marker["x"], marker["y"]),
                        "size": (marker["w"], marker["h"]),
                        "confirmed": True
                    }
                    found_markers.append(marker_info)
                    print(f"  → Marker ID: {marker['info']} at position ({marker['x']}, {marker['y']})")
            
            # Check for red objects
            red_objects = self.detect_red_objects()
            if red_objects:
                print(f"Found {len(red_objects)} red objects at {direction_name}")
                for red_obj in red_objects:
                    # Try to focus gimbal on the red object for detailed inspection
                    self._focus_on_object(red_obj, yaw_angle)
                    
                    # Check again for markers after focusing
                    time.sleep(0.3)
                    focused_markers = self.get_markers(max_age=0.2)
                    
                    if focused_markers:
                        for marker in focused_markers:
                            marker_info = {
                                "direction": direction_name,
                                "yaw": yaw_angle,
                                "id": str(marker["info"]),
                                "position": (marker["x"], marker["y"]),
                                "size": (marker["w"], marker["h"]),
                                "confirmed": True,
                                "was_red_object": True
                            }
                            found_markers.append(marker_info)
                            print(f"  → Confirmed Marker ID: {marker['info']} from red object")
                    else:
                        print(f"  → Red object detected but no valid marker found")
        
        # Additional corner scanning for markers in 90° corners
        corner_positions = [("Corner LF", -45), ("Corner RF", 45)]
        
        for direction_name, yaw_angle in corner_positions:
            print(f"\n--- Scanning {direction_name} (Yaw: {yaw_angle}°) ---")
            
            self.ep_gimbal.moveto(pitch=0, yaw=yaw_angle, yaw_speed=180).wait_for_completed()
            time.sleep(0.5)
            
            markers = self.get_markers(max_age=0.3)
            red_objects = self.detect_red_objects()
            
            if markers or red_objects:
                if markers:
                    for marker in markers:
                        marker_info = {
                            "direction": direction_name,
                            "yaw": yaw_angle,
                            "id": str(marker["info"]),
                            "position": (marker["x"], marker["y"]),
                            "size": (marker["w"], marker["h"]),
                            "confirmed": True
                        }
                        found_markers.append(marker_info)
                        print(f"  → Corner Marker ID: {marker['info']}")
                
                if red_objects and not markers:
                    for red_obj in red_objects:
                        self._focus_on_object(red_obj, yaw_angle)
                        time.sleep(0.3)
                        focused_markers = self.get_markers(max_age=0.2)
                        
                        if focused_markers:
                            for marker in focused_markers:
                                marker_info = {
                                    "direction": direction_name,
                                    "yaw": yaw_angle,
                                    "id": str(marker["info"]),
                                    "position": (marker["x"], marker["y"]),
                                    "size": (marker["w"], marker["h"]),
                                    "confirmed": True,
                                    "was_red_object": True
                                }
                                found_markers.append(marker_info)
                                print(f"  → Corner Marker ID: {marker['info']} from red object")
        
        # Return gimbal to center
        self.ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=180).wait_for_completed()
        
        if found_markers:
            print(f"\n=== TOTAL FOUND: {len(found_markers)} markers ===")
            for marker in found_markers:
                print(f"  ID: {marker['id']} at {marker['direction']}")
        else:
            print("\n=== NO MARKERS FOUND ===")
            
        return found_markers

    def _focus_on_object(self, red_obj, base_yaw):
        """Fine-tune gimbal to focus on detected red object"""
        # Calculate offset from center of frame (assuming 640x480 resolution)
        frame_center_x = 320
        center_offset = red_obj["x"] - frame_center_x
        
        # Convert pixel offset to angle offset (rough estimation)
        angle_per_pixel = 0.1  # Adjust based on camera FOV
        angle_offset = center_offset * angle_per_pixel
        
        target_yaw = base_yaw + angle_offset
        target_yaw = max(-90, min(90, target_yaw))  # Clamp to gimbal limits
        
        self.ep_gimbal.moveto(pitch=0, yaw=target_yaw, yaw_speed=90).wait_for_completed()
        time.sleep(0.2)

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
_fig, _ax = plt.subplots(figsize=(10, 10))


def plot_maze(current_cell, visited, walls, path_stack, detected_markers=None, title="Real-time Maze Exploration"):
    """Enhanced plotting function that displays markers on walls"""
    ax = _ax
    ax.clear()
    
    # Draw visited cells
    for x, y in visited:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightgray', edgecolor='gray'))
    
    # Draw walls with different styles
    for wall, style in walls.items():
        (x1, y1), (x2, y2) = wall
        line_style = '--' if style == 'dashed' else '-'
        color = 'r' if style == 'dashed' else 'k'
        
        if x1 == x2:  # Horizontal wall
            ax.plot([x1 - 0.5, x1 + 0.5], [max(y1, y2) - 0.5, max(y1, y2) - 0.5], 
                   color=color, linestyle=line_style, lw=4)
        else:  # Vertical wall
            ax.plot([max(x1, x2) - 0.5, max(x1, x2) - 0.5], [y1 - 0.5, y1 + 0.5], 
                   color=color, linestyle=line_style, lw=4)

    # Draw path
    if len(path_stack) > 1:
        path_x, path_y = zip(*path_stack)
        ax.plot(path_x, path_y, 'b-o', markersize=5)
    
    # Draw detected markers on walls
    if detected_markers:
        for cell_pos, markers_at_cell in detected_markers.items():
            for marker_info in markers_at_cell:
                wall_pos = marker_info['wall_position']
                marker_id = marker_info['id']
                
                # Draw red square marker symbol with number
                ax.add_patch(plt.Rectangle((wall_pos[0] - 0.1, wall_pos[1] - 0.1), 0.2, 0.2, 
                                         facecolor='red', edgecolor='darkred', linewidth=2))
                ax.text(wall_pos[0], wall_pos[1], str(marker_id), 
                       fontsize=12, fontweight='bold', color='white', 
                       ha='center', va='center')
    
    # Draw current robot position
    cx, cy = current_cell
    ax.plot(cx, cy, 'ro', markersize=12, label='Robot')
    
    # Set plot limits and formatting
    all_x = [c[0] for c in visited] or [0]
    all_y = [c[1] for c in visited] or [0]
    ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
    ax.set_ylim(min(all_y) - 1.5, max(all_y) + 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend()
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
        self.walls = {}
        self.current_orientation = self._get_discretized_orientation(self.ctrl.get_yaw_deg())
        
        # NEW: Storage for detected markers
        self.detected_markers = {}  # {cell_position: [marker_info, ...]}

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

    def _get_wall_center_position(self, current_cell, direction_name, yaw_angle):
        """Calculate the center position of a wall for marker placement"""
        cx, cy = current_cell
        
        if direction_name == "Left":
            return (cx - 0.5, cy)
        elif direction_name == "Right":
            return (cx + 0.5, cy)  
        elif direction_name == "Front":
            if self.current_orientation == 0:  # North
                return (cx, cy + 0.5)
            elif self.current_orientation == 1:  # East
                return (cx + 0.5, cy)
            elif self.current_orientation == 2:  # South
                return (cx, cy - 0.5)
            elif self.current_orientation == 3:  # West
                return (cx - 0.5, cy)
        elif direction_name.startswith("Corner"):
            if "LF" in direction_name:  # Left-Front corner
                return (cx - 0.35, cy + 0.35)
            elif "RF" in direction_name:  # Right-Front corner  
                return (cx + 0.35, cy + 0.35)
        
        return (cx, cy)  # Default fallback

    def scan_and_align(self):
        d = self.ctrl.eye()
        l_cm, f_cm, r_cm = d.get("L"), d.get("F"), d.get("R")
        print(f"Scan distances (cm): L={l_cm}, F={f_cm}, R={r_cm}")

        # Check if outside map
        is_outside = False
        is_l_open = l_cm is not None and l_cm > self.WALL_THRESHOLD
        is_f_open = f_cm is not None and f_cm > self.WALL_THRESHOLD
        is_r_open = r_cm is not None and r_cm > self.WALL_THRESHOLD
        
        if is_l_open and is_f_open and is_r_open:
            current_width_m = (l_cm + r_cm) / 100.0
            if current_width_m >= self.MAX_MAZE_WIDTH_M:
                is_outside = True
        
        # Alignment logic (same as before)
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

        if f is not None and not is_outside:
            forward_err = f - self.TARGET_DISTANCE_M
            if abs(forward_err) > self.TOLERANCE_M:
                step = float(np.clip(abs(forward_err) / 2.0, 0.0, self.MAX_FORWARD_STEP_M))
                if forward_err > 0: moves["backward"] = step
                else: moves["forward"] = step

        if moves["slide_left"] > 0: self.ctrl.slide_left(moves["slide_left"])
        elif moves["slide_right"] > 0: self.ctrl.slide_right(moves["slide_right"])
        if moves["backward"] > 0: self.ctrl.move_backward(moves["backward"])
        elif moves["forward"] > 0: self.ctrl.move_forward(moves["forward"])

        self.ctrl.ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=180).wait_for_completed()
        print(f"Alignment moves (m): {moves}")
        
        return d, is_outside

    def explore(self):
        print("Starting DFS Maze Solver with Enhanced Marker Detection...")
        while self.path_stack:
            current_cell = self.path_stack[-1]

            # Scan and align
            distances, is_outside = self.scan_and_align()

            # Enhanced marker detection
            print(f"\n🔍 Performing enhanced marker scan at {current_cell}")
            found_markers = self.ctrl.scan_for_markers_and_red_objects()
            
            # Process and store found markers
            if found_markers:
                if current_cell not in self.detected_markers:
                    self.detected_markers[current_cell] = []
                
                for marker in found_markers:
                    wall_pos = self._get_wall_center_position(
                        current_cell, marker['direction'], marker['yaw']
                    )
                    
                    marker_info = {
                        'id': marker['id'],
                        'direction': marker['direction'],
                        'wall_position': wall_pos,
                        'detection_details': marker
                    }
                    
                    # Avoid duplicates
                    existing_ids = [m['id'] for m in self.detected_markers[current_cell]]
                    if marker['id'] not in existing_ids:
                        self.detected_markers[current_cell].append(marker_info)
                        print(f"📍 Stored marker {marker['id']} at {marker['direction']} wall")

            # Update plot with markers
            plot_maze(current_cell, self.visited, self.walls, self.path_stack, 
                     self.detected_markers, f"Maze Exploration - Cell {current_cell}")

            print(f"\nPosition: {current_cell}, Orientation: {self.current_orientation} (Yaw: {self.ctrl.get_yaw_deg():.1f}°)")

            # Map the current cell
            if current_cell not in self.maze_map:
                self._scan_and_map(current_cell, distances, is_outside)

            # Handle out-of-bounds
            if is_outside:
                print("[Action] Out of bounds detected. Backtracking to re-enter map.")
                self._backtrack()
                continue

            # Try to move to next unvisited cell
            if self._find_and_move_to_next_cell(current_cell):
                continue

            # Backtrack if no unvisited neighbors
            if not self._backtrack():
                break

        print("\nDFS exploration complete.")
        
        # Final summary
        total_markers = sum(len(markers) for markers in self.detected_markers.values())
        print(f"\n📊 EXPLORATION SUMMARY:")
        print(f"   • Total cells visited: {len(self.visited)}")
        print(f"   • Total markers found: {total_markers}")
        
        if self.detected_markers:
            print(f"   • Marker details:")
            for cell_pos, markers in self.detected_markers.items():
                marker_ids = [m['id'] for m in markers]
                print(f"     Cell {cell_pos}: IDs {marker_ids}")
        
        plot_maze(self.path_stack[-1], self.visited, self.walls, self.path_stack,
                 self.detected_markers, "Final Maze Map with Detected Markers")
        finalize_show()

    def _scan_and_map(self, cell, distances=None, is_outside=False):
        if distances is None:
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

            if is_open_by_dist and is_outside and move_key in ['L', 'R']:
                # Out-of-bounds opening -> dashed wall
                self.walls[wall_tuple] = 'dashed'
            elif is_open_by_dist:
                # Normal opening
                open_directions.add(direction)
            else:
                # Solid wall
                self.walls[wall_tuple] = 'solid'

        self.maze_map[cell] = open_directions
        print(f"Mapped {cell} with open directions: {sorted(list(open_directions))}")
        print(f"Walls updated: {len(self.walls)} total walls.")

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
            self.path_stack.append(current_cell)
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
        print("Robot connected. Initializing enhanced solver...")
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