import robomaster
from robomaster import robot
import time
import math
import matplotlib.pyplot as plt
import numpy as np

# --- Global Variables ---
last_distance = None  # Last distance read by the sensor
current_x = 0
current_y = 0
current_yaw = 0.0

# --- Maze and Mapping Variables ---
maze_map = {}  # Stores the discovered maze layout. Key: (x,y), Value: {open_directions}
visited = set([(0, 0)]) # A set to store visited grid coordinates
path_stack = [(0, 0)]  # Stack to store the path of (grid_x, grid_y) cells
walls = set() # A set to store detected walls for plotting

# --- Constants ---
WALL_THRESHOLD = 60   # Distance in cm to consider a path blocked (a wall)
CELL_SIZE = 0.6       # The size of a grid cell in the maze in meters (e.g., 60cm)

# --- Plotting Setup ---
plt.ion() # Turn on interactive mode for live plotting
fig, ax = plt.subplots(figsize=(8, 8))

# --- Sensor and Position Callbacks ---

def sub_data_handler(sub_info):
    """Callback function to handle distance sensor data."""
    global last_distance
    # The sensor returns distance in millimeters, converting to centimeters.
    last_distance = int(sub_info[0]) / 10

def sub_attitude_handler(attitude_info):
    """Callback function to handle chassis attitude (orientation) data."""
    global current_yaw
    yaw, pitch, roll = attitude_info
    current_yaw = yaw

# --- Hardware Interaction Functions ---

def read_distance_at(yaw_angle):
    """Rotates the gimbal to a specific angle and reads the distance."""
    global last_distance
    last_distance = None
    ep_gimbal.moveto(pitch=0, yaw=yaw_angle, yaw_speed=180).wait_for_completed()
    ep_sensor.sub_distance(freq=20, callback=sub_data_handler)

    start_time = time.time()
    while last_distance is None and (time.time() - start_time) < 1.0:
        time.sleep(0.05)
    ep_sensor.unsub_distance()
    return last_distance

def eye():
    """Reads distances from three directions: Right, Front, and Left."""
    print("Scanning: [F, L, R]")
    # เรียงลำดับการสแกนใหม่เพื่อลดการหมุนของ Gimbal ที่ไม่จำเป็น
    distan = {
        "F": read_distance_at(0),
        "L": read_distance_at(-90),
        "R": read_distance_at(90)
    }
    ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=180).wait_for_completed()
    return distan

# --- Movement and Navigation Functions ---

def move_forward():
    """Moves the robot forward by one CELL_SIZE."""
    print(f"Action: Moving forward {CELL_SIZE}m")
    ep_chassis.move(x=CELL_SIZE, y=0, z=0, xy_speed=0.5).wait_for_completed()
    time.sleep(1)

def turn(angle):
    """Turns the robot by a specific angle in degrees."""
    # Note: Robomaster z-axis turn is counter-clockwise positive.
    # We send a negative angle to turn clockwise for a positive input.
    print(f"Action: Turning {angle} degrees")
    ep_chassis.move(x=0, y=0, z=-angle, z_speed=45).wait_for_completed()
    time.sleep(1)

def get_discretized_orientation():
    """Converts continuous yaw angle to one of four discrete directions."""
    # 0: North (+x), 1: East (+y), 2: South (-x), 3: West (-y)
    # Note: Robomaster coordinate system has +x forward, +y to the left.
    # To match a standard grid (x right, y up), we can map:
    # Robomaster Yaw 0   -> Direction 0 (North, +Y)
    # Robomaster Yaw 90  -> Direction 3 (West, -X)
    # Robomaster Yaw -90 -> Direction 1 (East, +X)
    # Robomaster Yaw 180 -> Direction 2 (South, -Y)
    if -45 <= current_yaw < 45:         return 0  # North
    elif 45 <= current_yaw < 135:       return 3  # West
    elif -135 > current_yaw >= -180 or 135 <= current_yaw <= 180: return 2  # South
    elif -135 <= current_yaw < -45:     return 1  # East

def get_target_coordinates(grid_x, grid_y, direction):
    """Calculates the coordinates of the next cell based on direction."""
    if direction == 0: return grid_x, grid_y + 1   # North
    elif direction == 1: return grid_x + 1, grid_y   # East
    elif direction == 2: return grid_x, grid_y - 1   # South
    elif direction == 3: return grid_x - 1, grid_y   # West
    return grid_x, grid_y

def get_relative_directions(robot_orientation):
    """Gets the global directions for Front, Right, and Left relative to robot."""
    front = robot_orientation
    right = (robot_orientation + 1) % 4
    left = (robot_orientation - 1 + 4) % 4
    return {"F": front, "R": right, "L": left}

# --- Plotting Function ---

def plot_maze(current_cell):
    """Plots the current state of the maze exploration."""
    ax.clear()
    
    # Plot visited cells
    for (x, y) in visited:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='lightgray', edgecolor='gray'))

    # Plot walls
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        # Vertical wall
        if x1 == x2:
            ax.plot([x1 - 0.5, x1 + 0.5], [max(y1, y2) - 0.5, max(y1, y2) - 0.5], 'k-', linewidth=4)
        # Horizontal wall
        else:
            ax.plot([max(x1, x2) - 0.5, max(x1, x2) - 0.5], [y1 - 0.5, y1 + 0.5], 'k-', linewidth=4)

    # Plot the path taken
    if len(path_stack) > 1:
        path_x, path_y = zip(*path_stack)
        ax.plot(path_x, path_y, 'b-o', markersize=5, label='Path')

    # Highlight the current robot position
    cx, cy = current_cell
    ax.plot(cx, cy, 'ro', markersize=12, label='Robot')

    # Set plot limits and labels
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


# --- DFS Maze Solving Logic ---

def dfs_maze_solver():
    """Solves a maze using Depth-First Search with map memorization."""
    print("Starting DFS Maze Solver...")

    while path_stack:
        current_cell = path_stack[-1]
        current_grid_x, current_grid_y = current_cell
        print(f"\nAt cell: {current_cell}. Path stack size: {len(path_stack)}")

        # Update the live plot
        plot_maze(current_cell)

        robot_orientation = get_discretized_orientation()
        
        # --- Check if cell is already mapped ---
        if current_cell not in maze_map:
            print(f"Cell {current_cell} is unmapped. Scanning environment...")
            distances = eye()
            print(f"Sensor readings: {distances}")
            
            relative_dirs = get_relative_directions(robot_orientation)
            open_directions = set()
            
            # Check for open paths and update walls set
            for move, direction in relative_dirs.items():
                neighbor = get_target_coordinates(current_grid_x, current_grid_y, direction)
                dist = distances.get(move)
                if dist is not None and dist > WALL_THRESHOLD:
                    open_directions.add(direction)
                else:
                    # Add a wall between current and neighbor cell for plotting
                    wall = tuple(sorted((current_cell, neighbor)))
                    walls.add(wall)

            maze_map[current_cell] = open_directions
            print(f"Mapped cell {current_cell} with open directions: {open_directions}")
        
        else:
            print(f"Cell {current_cell} already mapped. Using cached data.")
            open_directions = maze_map[current_cell]

        # --- Find an unvisited neighboring cell ---
        next_move_direction = -1
        next_cell = None
        
        # Prioritize exploring new paths
        for direction in open_directions:
            target_cell = get_target_coordinates(current_grid_x, current_grid_y, direction)
            if target_cell not in visited:
                next_move_direction = direction
                next_cell = target_cell
                break

        if next_cell:
            # --- Move to the next cell ---
            print(f"Found unvisited neighbor at {next_cell}. Moving...")
            
            # 1. Calculate turn angle relative to robot's current orientation
            # This logic needs careful mapping from global direction to robot's relative turn
            # Assuming our orientation mapping is correct
            # Standard turn angle mapping: (target_dir - current_dir) * 90
            turn_angle = (next_move_direction - robot_orientation) * 90
            
            # Normalize angle to [-180, 180]
            if turn_angle > 180: turn_angle -= 360
            if turn_angle < -180: turn_angle += 360
            
            turn(turn_angle)

            # 2. Move forward into the new cell
            move_forward()

            # 3. Update state
            visited.add(next_cell)
            path_stack.append(next_cell)

        else:
            # --- Backtrack: No unvisited neighbors ---
            print("Dead end or all neighbors visited. Backtracking...")
            path_stack.pop()

            if not path_stack:
                # This is the final exit condition
                print("\nReturned to start and all reachable cells explored. Halting.")
                break 

            # Determine direction to previous cell and move
            prev_cell = path_stack[-1]
            
            # Turn 180 degrees to face back the way we came
            turn(180)
            
            # Move forward to the previous cell
            move_forward()
    
    print("\nDFS exploration complete.")
    # Show the final map
    print("Displaying final map...")
    plt.ioff()
    plot_maze((0,0)) # Plot final state
    plt.show()


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_chassis = ep_robot.chassis
    ep_gimbal = ep_robot.gimbal
    ep_sensor = ep_robot.sensor

    # Reset gimbal to center position
    ep_gimbal.recenter().wait_for_completed()
    
    # Subscribe to attitude to always know the robot's orientation
    ep_chassis.sub_attitude(freq=10, callback=sub_attitude_handler)
    
    # Give a moment for the subscription to start and for the robot to stabilize
    time.sleep(2)
    print("Robot Initialized. Starting maze solver in 3 seconds...")
    time.sleep(3)

    try:
        # Start the maze solving algorithm
        dfs_maze_solver()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Unsubscribe and close connection
        print("Cleaning up and closing connection.")
        ep_chassis.unsub_attitude()
        ep_robot.close()