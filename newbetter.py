import time
import math
import numpy as np
import matplotlib.pyplot as plt
from robomaster import robot, vision
import threading
from typing import List, Dict, Tuple, Set, Optional

# ===================== PID Controller Class =====================
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self._prev_error, self._integral = 0.0, 0.0
        self._last_time = time.time()

    def compute(self, current_value):
        t, dt = time.time(), time.time() - self._last_time
        if dt <= 0: return 0.0
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

        # State
        self.last_distance_cm = None
        self.current_x, self.current_y, self.current_yaw = 0.0, 0.0, 0.0

        # Callbacks
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
        self.ep_chassis.sub_attitude(freq=10, callback=_att_cb)
        self.ep_chassis.sub_position(freq=50, callback=_pos_cb)
        self._dist_subscribed = False
        self._dist_cb = _dist_cb

        # Vision Initialization
        self.ep_camera = self.ep_robot.camera
        self.ep_vision = self.ep_robot.vision
        self._markers = []
        self._markers_lock = threading.Lock()

        def _on_markers(marker_info):
            now = time.time()
            with self._markers_lock:
                # กรองเฉพาะ marker ที่เป็นตัวเลข 0-9
                self._markers = [
                    {"x": x, "y": y, "w": w, "h": h, "info": info, "ts": now}
                    for (x, y, w, h, info) in marker_info 
                    if str(info).isdigit()  # ตรวจสอบว่า info เป็นตัวเลขหรือไม่
                ]

        self.ep_camera.start_video_stream(display=False)
        self.ep_vision.sub_detect_info(name="marker", callback=_on_markers)
        time.sleep(1.0)

    def get_markers(self, max_age=0.6) -> List[Dict]:
        now = time.time()
        with self._markers_lock:
            return [m for m in self._markers if now - m["ts"] <= max_age]

    # Movement methods (same as original)
    def get_yaw_deg(self) -> float: 
        return self.current_yaw
        
    def get_xy_m(self) -> Tuple[float, float]: 
        return self.current_x, self.current_y

    def _sub_distance(self, freq=20):
        if not self._dist_subscribed:
            self.ep_sensor.sub_distance(freq=freq, callback=self._dist_cb)
            self._dist_subscribed = True

    def _unsub_distance(self):
        if self._dist_subscribed:
            self.ep_sensor.unsub_distance()
            self._dist_subscribed = False

    def read_distance_at(self, yaw_angle_deg, samples=5, timeout_s=1.0) -> Optional[float]:
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
        if not distances: return None
        med = float(np.median(distances))
        print(f"Median distance at yaw {yaw_angle_deg}: {med:.1f} cm")
        return med

    def eye(self) -> Dict[str, Optional[float]]:
        print("Scanning: [L, F, R]")
        dist = {
            "L": self.read_distance_at(-90), 
            "F": self.read_distance_at(0), 
            "R": self.read_distance_at(90)
        }
        self.ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=180).wait_for_completed()
        return dist

    def slide_left(self, distance_m: float):
        print(f"Action: Sliding left {distance_m:.2f} m")
        self.ep_chassis.move(x=0, y=-distance_m, z=0).wait_for_completed()

    def slide_right(self, distance_m: float):
        print(f"Action: Sliding right {distance_m:.2f} m")
        self.ep_chassis.move(x=0, y=distance_m, z=0).wait_for_completed()

    def move_forward(self, distance_m: float):
        print(f"Action: Moving forward {distance_m:.2f} m")
        self.ep_chassis.move(x=distance_m, y=0, z=0).wait_for_completed()

    def move_backward(self, distance_m: float):
        print(f"Action: Moving backward {distance_m:.2f} m")
        self.ep_chassis.move(x=-distance_m, y=0, z=0).wait_for_completed()

    def stop(self):
        self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.2)
        time.sleep(0.2)

    def turn(self, angle_deg: float):
        print(f"Action: Turning {angle_deg:.1f} degrees")
        self.ep_chassis.move(x=0, y=0, z=-angle_deg, z_speed=45).wait_for_completed()
        time.sleep(0.5)

    def move_forward_pid(self, cell_size_m: float, Kp=1.2, Ki=0.05, Kd=0.1, v_clip=0.7, tol_m=0.001):
        print(f"Action: Moving forward {cell_size_m} m")
        pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=cell_size_m)
        sx, sy = self.get_xy_m()
        while True:
            dist = math.hypot(self.current_x - sx, self.current_y - sy)
            speed = float(np.clip(pid.compute(dist), -v_clip, v_clip))
            self.ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=0.1)
            if abs(cell_size_m - dist) < tol_m: break
            time.sleep(0.02)
        self.stop()
        print("Movement complete.")

    def close(self):
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
        except Exception as e: print(f"Error during cleanup: {e}")

# ===================== Plotting Functions =====================
plt.ion()
_fig, _ax = plt.subplots(figsize=(8, 8))

def plot_maze(current_cell, visited, walls, path_stack, markers=None, title="Real-time Maze Exploration"):
    """
    Plot the maze with markers and current robot position
    
    Args:
        current_cell: (x,y) tuple of current robot cell
        visited: set of visited (x,y) cells
        walls: set of wall segments ((x1,y1), (x2,y2))
        path_stack: list of (x,y) cells in current path
        markers: list of marker dicts with 'x', 'y', 'info' keys
        title: plot title
    """
    ax = _ax
    ax.clear()
    
    # Draw visited cells
    for (x, y) in visited:
        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, 
                                 facecolor='lightgray', edgecolor='gray'))
    
    # Draw walls
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        if x1 == x2:  # Vertical wall
            y_center = (y1 + y2) / 2
            ax.plot([x1-0.5, x1+0.5], [y_center, y_center], 'k-', lw=4)
        else:  # Horizontal wall
            x_center = (x1 + x2) / 2
            ax.plot([x_center, x_center], [y1-0.5, y1+0.5], 'k-', lw=4)
    
    # Draw path
    if len(path_stack) > 1:
        path_x, path_y = zip(*path_stack)
        ax.plot(path_x, path_y, 'b-o', markersize=5, linewidth=2, alpha=0.5)
    
    # Draw markers if any
    if markers:
        for marker in markers:
            # Convert from image coordinates to maze coordinates
            # Assuming marker x is in [0,1] where 0=left, 1=right of robot
            # and robot is at current_cell facing its current orientation
            # This is a simplified approach - adjust based on your coordinate system
            marker_x = current_cell[0] + (marker['x'] - 0.5) * 0.5  # Scale and center
            marker_y = current_cell[1] + 0.5  # Place in front of robot
            
            # ตรวจสอบว่า marker เป็นตัวเลข 0-9 หรือไม่
            marker_info = str(marker.get('info', ''))
            if marker_info.isdigit():
                ax.plot(marker_x, marker_y, 'rs', markersize=12, alpha=0.7)  # เปลี่ยนเป็นสี่เหลี่ยมสีแดง
                ax.text(marker_x, marker_y, marker_info, 
                       ha='center', va='center', color='white', weight='bold', fontsize=10)
    
    # Draw robot position
    cx, cy = current_cell
    ax.plot(cx, cy, 'ro', markersize=12, label='Robot')
    
    # Adjust plot limits
    all_x = [c[0] for c in visited] or [0]
    all_y = [c[1] for c in visited] or [0]
    ax.set_xlim(min(all_x)-1.5, max(all_x)+1.5)
    ax.set_ylim(min(all_y)-1.5, max(all_y)+1.5)
    
    # Final touches
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_title(title)
    plt.pause(0.1)

def finalize_show():
    """Finalize and show the plot (call at the end of exploration)"""
    plt.ioff()
    plt.show()

# ===================== MazeSolver Class =====================
class MazeSolver:
    WALL_THRESHOLD = 60  # cm
    CELL_SIZE = 0.6  # meters
    TARGET_DISTANCE_M = 0.04  # ~5 cm from wall
    TOLERANCE_M = 0.02  # ±2 cm tolerance
    MAX_LATERAL_STEP_M = 0.05  # Max slide distance per step
    MAX_FORWARD_STEP_M = 0.05  # Max forward/backward distance per step

    def __init__(self, ctrl: Control):
        self.ctrl = ctrl
        self.maze_map = {}  # cell -> set of open directions
        self.visited = set([(0, 0)])  # Visited cells
        self.path_stack = [(0, 0)]  # Current path
        self.walls = set()  # Wall segments ((x1,y1), (x2,y2))
        self.markers = []  # List of detected markers with positions
        self.current_orientation = self._get_discretized_orientation(self.ctrl.get_yaw_deg())

    @staticmethod
    def _get_discretized_orientation(yaw_deg):
        yaw = (yaw_deg + 360) % 360
        if yaw >= 315 or yaw < 45:   return 0  # North
        elif 45 <= yaw < 135:  return 3  # East
        elif 135 <= yaw < 225: return 2  # South
        else: return 1  # West

    @staticmethod
    def _get_target_coordinates(grid_x, grid_y, direction):
        if direction == 0:   return (grid_x, grid_y + 1)  # North
        elif direction == 1: return (grid_x - 1, grid_y)  # West
        elif direction == 2: return (grid_x, grid_y - 1)  # South
        elif direction == 3: return (grid_x + 1, grid_y)  # East

    @staticmethod
    def _get_relative_directions(orientation):
        return {"L": (orientation - 1) % 4, 
                "F": orientation, 
                "R": (orientation + 1) % 4}

    @staticmethod
    def _get_direction_to_neighbor(current_cell, target_cell):
        dx = target_cell[0] - current_cell[0]
        dy = target_cell[1] - current_cell[1]
        if dx == 1: return 3  # East
        if dx == -1: return 1  # West
        if dy == 1: return 0  # North
        if dy == -1: return 2  # South
        return None

    def scan_and_align(self) -> Dict[str, Optional[float]]:
        """
        Scan environment and align robot in the corridor.
        Returns distance measurements in cm.
        """
        d = self.ctrl.eye()  # Get distances in cm
        l_cm, f_cm, r_cm = d.get("L"), d.get("F"), d.get("R")
        print(f"Scan distances (cm): L={l_cm}, F={f_cm}, R={r_cm}")

        # Convert to meters, None if measurement is invalid
        l = None if l_cm is None else l_cm / 100.0
        f = None if f_cm is None else f_cm / 100.0
        r = None if r_cm is None else r_cm / 100.0

        moves = {"slide_left": 0.0, "slide_right": 0.0, 
                "forward": 0.0, "backward": 0.0}

        # Center in corridor if both sides visible
        if l is not None and r is not None:
            lateral_err = l - r
            if abs(lateral_err) > self.TOLERANCE_M:
                step = min(abs(lateral_err) / 2.0, self.MAX_LATERAL_STEP_M)
                if lateral_err > 0:
                    moves["slide_left"] = step
                else:
                    moves["slide_right"] = step
        else:
            # Only one side visible, maintain distance from it
            if l is not None:
                e = self.TARGET_DISTANCE_M - l
                if abs(e) > self.TOLERANCE_M:
                    if e > 0:  moves["slide_right"] = min(e, self.MAX_LATERAL_STEP_M)
                    else:      moves["slide_left"] = min(-e, self.MAX_LATERAL_STEP_M)
            if r is not None:
                e = self.TARGET_DISTANCE_M - r
                if abs(e) > self.TOLERANCE_M:
                    if e > 0:  moves["slide_left"] = max(moves["slide_left"], min(e, self.MAX_LATERAL_STEP_M))
                    else:      moves["slide_right"] = max(moves["slide_right"], min(-e, self.MAX_LATERAL_STEP_M))

        # Adjust distance from front wall
        if f is not None:
            e = self.TARGET_DISTANCE_M - f
            if abs(e) > self.TOLERANCE_M:
                if e > 0:  moves["backward"] = min(e, self.MAX_FORWARD_STEP_M)
                else:      moves["forward"] = min(-e, self.MAX_FORWARD_STEP_M)

        # Execute movements
        if moves["slide_left"] > 0:   self.ctrl.slide_left(moves["slide_left"])
        elif moves["slide_right"] > 0: self.ctrl.slide_right(moves["slide_right"])

        if moves["backward"] > 0:     self.ctrl.move_backward(moves["backward"])
        elif moves["forward"] > 0:    self.ctrl.move_forward(moves["forward"])

        # Recenter gimbal
        self.ctrl.ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=180).wait_for_completed()

        print(f"Alignment moves (m): {moves}")
        return d  # Return distances in cm

    def explore(self):
        """Main exploration loop with marker detection and display"""
        print("Starting DFS Maze Solver with Marker Detection...")
        
        while self.path_stack:
            current_cell = self.path_stack[-1]
            
            # Scan environment and align
            distances = self.scan_and_align()
            
            # Get detected markers
            markers = self.ctrl.get_markers(max_age=0.6)
            
            # Store markers with their positions
            for marker in markers:
                marker_pos = self._calculate_marker_position(current_cell, marker)
                if marker_pos:
                    self.markers.append({
                        'x': marker_pos[0],
                        'y': marker_pos[1],
                        'info': marker.get('info', '?')
                    })
            
            # Update plot with current state and markers
            plot_maze(
                current_cell=current_cell,
                visited=self.visited,
                walls=self.walls,
                path_stack=self.path_stack,
                markers=self.markers,
                title=f"Exploring - Markers: {len(self.markers)}"
            )
            
            print(f"Position: {current_cell}, Orientation: {self.current_orientation} (Yaw: {self.ctrl.get_yaw_deg():.1f}°)")
            
            # Log detected markers
            if markers:
                ids = [str(m.get('info', '?')) for m in markers]
                print(f"[Marker] Detected {len(markers)} markers - IDs: {ids}")
            
            # If current cell not mapped, scan and map it
            if current_cell not in self.maze_map:
                self._scan_and_map(current_cell, distances)
            
            # Try to find and move to next unvisited cell
            if self._find_and_move_to_next_cell(current_cell):
                continue
            
            # If no unvisited cells, backtrack
            if not self._backtrack():
                break
        
        print("\nExploration complete!")
        plot_maze(
            current_cell=self.path_stack[-1] if self.path_stack else (0, 0),
            visited=self.visited,
            walls=self.walls,
            path_stack=self.path_stack,
            markers=self.markers,
            title=f"Final Map - Visited: {len(self.visited)} cells, Markers: {len(self.markers)}"
        )
        finalize_show()
        self.ctrl.close()

    def _calculate_marker_position(self, current_cell, marker):
        """
        Calculate the absolute position of a marker based on robot's position and orientation.
        This is a simplified approach - adjust based on your coordinate system.
        """
        try:
            # Get marker position in image coordinates (normalized to [0,1])
            marker_x_img = marker.get('x', 0.5)  # 0=left, 1=right of image
            marker_y_img = marker.get('y', 0.5)  # 0=top, 1=bottom of image
            
            # Convert to maze coordinates
            # This is a simplified approach - adjust based on your needs
            marker_dist = 0.5  # Distance in front of robot (in cells)
            
            # Calculate position based on robot's orientation
            if self.current_orientation == 0:  # North
                return (current_cell[0], current_cell[1] + marker_dist)
            elif self.current_orientation == 1:  # West
                return (current_cell[0] - marker_dist, current_cell[1])
            elif self.current_orientation == 2:  # South
                return (current_cell[0], current_cell[1] - marker_dist)
            else:  # East
                return (current_cell[0] + marker_dist, current_cell[1])
                
        except Exception as e:
            print(f"Error calculating marker position: {e}")
            return None

    def _scan_and_map(self, cell, distances=None):
        """Scan and map the current cell"""
        if distances is None:
            print(f"Cell {cell} is unmapped. Scanning...")
            distances = self.ctrl.eye()
        
        relative_dirs = self._get_relative_directions(self.current_orientation)
        open_directions = set()

        # Check each direction (L/F/R)
        for move_key in ["L", "F", "R"]:
            direction = relative_dirs[move_key]
            dist_cm = distances.get(move_key)
            
            if dist_cm is not None and dist_cm > self.WALL_THRESHOLD:
                open_directions.add(direction)
            else:
                # Add wall to our map
                neighbor = self._get_target_coordinates(cell[0], cell[1], direction)
                self.walls.add(tuple(sorted((cell, neighbor))))
        
        self.maze_map[cell] = open_directions
        print(f"Mapped {cell} with open directions: {sorted(list(open_directions))}")

    def _find_and_move_to_next_cell(self, cell):
        """Find and move to an unvisited neighboring cell"""
        relative_dirs = self._get_relative_directions(self.current_orientation)
        
        # Check directions in order: Left, Forward, Right
        for move_key in ["L", "F", "R"]:
            direction = relative_dirs[move_key]
            if direction in self.maze_map.get(cell, set()):
                target_cell = self._get_target_coordinates(cell[0], cell[1], direction)
                if target_cell not in self.visited:
                    print(f"Moving {move_key} to {target_cell}...")
                    self._turn_to(direction)
                    self.ctrl.move_forward_pid(self.CELL_SIZE)
                    self.visited.add(target_cell)
                    self.path_stack.append(target_cell)
                    return True
        return False

    def _backtrack(self):
        """Backtrack to previous cell when no unvisited neighbors"""
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
        """Turn to face the target direction"""
        turn_angle = (target_direction - self.current_orientation) * 90
        
        # Normalize angle to [-180, 180]
        if turn_angle > 180: turn_angle -= 360
        if turn_angle < -180: turn_angle += 360
        
        if abs(turn_angle) > 1:  # Only turn if significant angle
            self.ctrl.stop()
            self.ctrl.turn(turn_angle)
            self.current_orientation = target_direction

# ===================== Main Execution =====================
if __name__ == "__main__":
    try:
        # Initialize controller and solver
        controller = Control(conn_type="ap")  # Use "ap" for WiFi, "sta" for direct connection
        solver = MazeSolver(controller)
        
        # Start exploration
        solver.explore()
        
    except KeyboardInterrupt:
        print("\nExploration interrupted by user.")
    except Exception as e:
        print(f"Error during exploration: {e}")
    finally:
        try:
            controller.close()
        except:
            pass
        print("Program terminated.")
        plt.ioff()
        plt.close('all')
