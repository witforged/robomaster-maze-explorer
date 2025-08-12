import robomaster
from robomaster import robot
import time
import math
import matplotlib.pyplot as plt
import numpy as np

# --- Global Variables ---
last_distance = None      # ระยะกำแพง
current_x = 0             # ตำแหน่งหุ่นแกน x (จาก Position Subscription)
current_y = 0             # ตำแหน่งหุ่นแกน y (จาก Position Subscription)
current_yaw = 0.0         # มุมหุ่น

# --- Maze and Mapping Variables ---
maze_map = {}
visited = set([(0, 0)])
path_stack = [(0, 0)]
walls = set()

# --- Constants ---
WALL_THRESHOLD = 60   # ระยะน้อยกว่า 60 มีกำแพง
CELL_SIZE = 0.6       # ขนาด grid (เมตร)

# --- Plotting Setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

# --- PID Controller Class ---
class PIDController:
    """A simple PID controller class."""
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self._prev_error = 0
        self._integral = 0
        self._last_time = time.time()

    def compute(self, current_value):
        """Calculates the PID output."""
        current_time = time.time()
        dt = current_time - self._last_time
        if dt == 0:
            return 0  # Avoid division by zero

        error = self.setpoint - current_value
        
        # Proportional term
        P_out = self.Kp * error
        
        # Integral term
        self._integral += error * dt
        I_out = self.Ki * self._integral
        
        # Derivative term
        derivative = (error - self._prev_error) / dt
        D_out = self.Kd * derivative
        
        # Total output
        output = P_out + I_out + D_out
        
        # Update state for next iteration
        self._prev_error = error
        self._last_time = current_time
        
        return output

# --- Sensor and Position Callbacks ---
def sub_data_handler(sub_info):      # ข้อมูล ToF
    global last_distance
    last_distance = int(sub_info[0]) / 10

def sub_attitude_handler(attitude_info):  # ข้อมูลมุมของหุ่น
    global current_yaw
    yaw, pitch, roll = attitude_info
    current_yaw = yaw

def sub_position_handler(position_info):
    """Callback function to handle chassis position data."""
    global current_x, current_y
    x, y, z = position_info
    current_x = x
    current_y = y

# --- Hardware Interaction Functions ---
def read_distance_at(yaw_angle):
    """Rotates the gimbal to a specific angle and reads the distance using a median filter."""
    global last_distance
    last_distance = None
    ep_gimbal.moveto(pitch=0, yaw=yaw_angle, yaw_speed=180).wait_for_completed()
    
    # เก็บค่าระยะทางหลายๆ ค่าเพื่อใช้ Median Filter
    distances = []
    ep_sensor.sub_distance(freq=20, callback=sub_data_handler)
    start_time = time.time()
    while len(distances) < 5 and (time.time() - start_time) < 1.0:  # รวบรวม 5 ค่าใน 1 วินาที
        if last_distance is not None:
            distances.append(last_distance)
            last_distance = None  # รีเซ็ตเพื่อรอค่าต่อไป
        time.sleep(0.05)  # รอ 50ms ต่อรอบ
    ep_sensor.unsub_distance()
    
    # ถ้ามีข้อมูลเพียงพอ คำนวณ Median
    if distances:
        median_distance = np.median(distances)
        print(f"Median distance at yaw {yaw_angle}: {median_distance:.1f} cm (from {distances})")
        return median_distance
    else:
        print(f"No valid distance readings at yaw {yaw_angle}")
        return None

def eye():
    """Scans distances in three directions: Front, Left, and Right."""
    print("Scanning: [F, L, R]")
    distan = {
        "F": read_distance_at(0),
        "L": read_distance_at(-90),
        "R": read_distance_at(90)
    }
    ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=180).wait_for_completed()
    return distan

def turn(angle):
    """Turns the robot by a specific angle in degrees."""
    print(f"Action: Turning {angle} degrees")
    ep_chassis.move(x=0, y=0, z=-angle, z_speed=45).wait_for_completed()
    time.sleep(1)

def move_forward_pid():
    """Moves the robot forward by CELL_SIZE using a PID controller for accuracy."""
    print(f"Action: Moving forward {CELL_SIZE}m using PID control.")
    
    pid = PIDController(Kp=1.2, Ki=0.05, Kd=0.1, setpoint=CELL_SIZE)
    
    start_x, start_y = current_x, current_y
    
    while True:
        distance_traveled = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
        speed = pid.compute(distance_traveled)
        speed = np.clip(speed, -0.7, 0.7)
        ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=0.1)
        error = CELL_SIZE - distance_traveled
        if abs(error) < 0.02:
            print("Target reached within tolerance.")
            break
        time.sleep(0.02)
    
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
    print("Movement complete.")
    time.sleep(1)

# --- Navigation and Plotting Functions ---
def get_discretized_orientation():
    if -45 <= current_yaw < 45: return 0
    elif 45 <= current_yaw < 135: return 3
    elif -135 > current_yaw >= -180 or 135 <= current_yaw <= 180: return 2
    elif -135 <= current_yaw < -45: return 1

def get_target_coordinates(grid_x, grid_y, direction):
    if direction == 0: return grid_x, grid_y + 1
    elif direction == 1: return grid_x + 1, grid_y
    elif direction == 2: return grid_x, grid_y - 1
    elif direction == 3: return grid_x - 1, grid_y
    return grid_x, grid_y

def get_relative_directions(robot_orientation):
    front = robot_orientation
    right = (robot_orientation + 1) % 4
    left = (robot_orientation - 1 + 4) % 4
    return {"F": front, "R": right, "L": left}

def plot_maze(current_cell):
    ax.clear()
    for (x, y) in visited:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightgray', edgecolor='gray'))
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        if x1 == x2:
            ax.plot([x1 - 0.5, x1 + 0.5], [max(y1, y2) - 0.5, max(y1, y2) - 0.5], 'k-', linewidth=4)
        else:
            ax.plot([max(x1, x2) - 0.5, max(x1, x2) - 0.5], [y1 - 0.5, y1 + 0.5], 'k-', linewidth=4)
    if len(path_stack) > 1:
        path_x, path_y = zip(*path_stack)
        ax.plot(path_x, path_y, 'b-o', markersize=5, label='Path')
    cx, cy = current_cell
    ax.plot(cx, cy, 'ro', markersize=12, label='Robot')
    all_x = [c[0] for c in visited]
    all_y = [c[1] for c in visited]
    ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
    ax.set_ylim(min(all_y) - 1.5, max(all_y) + 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(min(all_x) - 2, max(all_x) + 3, 1))
    ax.set_yticks(np.arange(min(all_y) - 2, max(all_y) + 3, 1))
    ax.grid(True)
    ax.set_title("Real-time Maze Exploration")
    plt.pause(0.1)

# --- DFS Logic ---
def dfs_maze_solver():
    print("Starting DFS Maze Solver...")
    while path_stack:
        current_cell = path_stack[-1]
        current_grid_x, current_grid_y = current_cell
        print(f"\nAt cell: {current_cell}. Path stack size: {len(path_stack)}")
        plot_maze(current_cell)
        robot_orientation = get_discretized_orientation()
        if current_cell not in maze_map:
            print(f"Cell {current_cell} is unmapped. Scanning environment...")
            distances = eye()
            print(f"Sensor readings: {distances}")
            relative_dirs = get_relative_directions(robot_orientation)
            open_directions = set()
            for move, direction in relative_dirs.items():
                neighbor = get_target_coordinates(current_grid_x, current_grid_y, direction)
                dist = distances.get(move)
                if dist is not None and dist > WALL_THRESHOLD:
                    open_directions.add(direction)
                else:
                    wall = tuple(sorted((current_cell, neighbor)))
                    walls.add(wall)
            maze_map[current_cell] = open_directions
            print(f"Mapped cell {current_cell} with open directions: {open_directions}")
        else:
            print(f"Cell {current_cell} already mapped. Using cached data.")
            open_directions = maze_map[current_cell]
        
        next_move_direction = -1
        next_cell = None
        for direction in open_directions:
            target_cell = get_target_coordinates(current_grid_x, current_grid_y, direction)
            if target_cell not in visited:
                next_move_direction = direction
                next_cell = target_cell
                break
        
        if next_cell:
            print(f"Found unvisited neighbor at {next_cell}. Moving...")
            turn_angle = (next_move_direction - robot_orientation) * 90
            if turn_angle > 180: turn_angle -= 360
            if turn_angle < -180: turn_angle += 360
            turn(turn_angle)
            move_forward_pid()
            visited.add(next_cell)
            path_stack.append(next_cell)
        else:
            print("Dead end or all neighbors visited. Backtracking...")
            path_stack.pop()
            if not path_stack:
                print("\nReturned to start and all reachable cells explored. Halting.")
                break
            turn(180)
            move_forward_pid()
    
    print("\nDFS exploration complete.")
    print("Displaying final map...")
    plt.ioff()
    plot_maze((0,0))
    plt.show()

# --- Main execution block ---
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_sensor = ep_robot.sensor

    ep_gimbal.recenter().wait_for_completed()
    
    ep_chassis.sub_attitude(freq=10, callback=sub_attitude_handler)
    ep_chassis.sub_position(freq=50, callback=sub_position_handler)
    
    time.sleep(2)
    
    try:
        dfs_maze_solver()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up and closing connection.")
        ep_chassis.unsub_attitude()
        ep_chassis.unsub_position()
        ep_robot.close()