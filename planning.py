# -*- coding:utf-8 -*-
"""
planing.py
- วางแผนด้วย DFS + ทำแผนที่
- เรียกใช้ Control (สแกน/หมุน/เดิน) และ plot
วิธีรัน: python planing.py
"""

import time
from control import Control     
import plot as mzplot               # โมดูลพล็อตที่เราเขียน

# ===== Constants (ตรงกับโค้ดต้นฉบับ) =====
WALL_THRESHOLD = 60    # cm: ระยะ <= 60 ถือว่ามีกำแพง
CELL_SIZE      = 0.6   # m: 1 ช่อง = 0.6 m

# ===== Global map states =====
maze_map = {}                 # {(grid_x, grid_y): set(directions_open)}
visited = set([(0, 0)])
path_stack = [(0, 0)]
walls = set()                 # set( ((x1,y1),(x2,y2)) )

# ===== Helpers for orientation & grid =====
def get_discretized_orientation(yaw_deg):
    """
    แปลง yaw จริง (deg) → ทิศดิสครีต 0..3
    0: North, 1: East, 2: South, 3: West
    ตรงตามนิยามเดิมในโค้ดของผู้ใช้
    """
    if -45 <= yaw_deg < 45:
        return 0  # North
    elif 45 <= yaw_deg < 135:
        return 3  # West
    elif -135 > yaw_deg >= -180 or 135 <= yaw_deg <= 180:
        return 2  # South
    elif -135 <= yaw_deg < -45:
        return 1  # East

def get_target_coordinates(grid_x, grid_y, direction):
    """คืนเพื่อนบ้านตามทิศ (นิยามเดิมของผู้ใช้)"""
    if direction == 0:   return grid_x,     grid_y + 1   # North
    elif direction == 1: return grid_x + 1, grid_y       # East
    elif direction == 2: return grid_x,     grid_y - 1   # South
    elif direction == 3: return grid_x - 1, grid_y       # West
    return grid_x, grid_y

def get_relative_directions(robot_orientation):
    """คืนทิศ global ของ F/R/L จากทิศปัจจุบันของหุ่น (0..3)"""
    front = robot_orientation
    right = (robot_orientation + 1) % 4
    left  = (robot_orientation - 1 + 4) % 4
    return {"F": front, "R": right, "L": left}

# ===== DFS main =====
def dfs_maze_solver(ctrl: Control):
    print("Starting DFS Maze Solver...")

    while path_stack:
        current_cell = path_stack[-1]
        current_grid_x, current_grid_y = current_cell

        # Plot
        mzplot.plot_maze(current_cell, visited, walls, path_stack)

        # Orientation
        yaw = ctrl.get_yaw_deg()
        robot_orientation = get_discretized_orientation(yaw)

        # ----- Mapping (scan if unmapped) -----
        if current_cell not in maze_map:
            print(f"\nAt cell: {current_cell} (unmapped). Scanning environment...")
            distances = ctrl.eye()   # dict: {"F":cm, "L":cm, "R":cm} or None
            print(f"Sensor readings: {distances}")

            relative_dirs = get_relative_directions(robot_orientation)
            open_directions = set()

            # อัปเดต open/wall
            for move, direction in relative_dirs.items():
                neighbor = get_target_coordinates(current_grid_x, current_grid_y, direction)
                dist_cm = distances.get(move)
                if (dist_cm is not None) and (dist_cm > WALL_THRESHOLD):
                    open_directions.add(direction)
                else:
                    # บันทึกกำแพงสำหรับการพล็อต
                    wall = tuple(sorted((current_cell, neighbor)))
                    walls.add(wall)

            maze_map[current_cell] = open_directions
            print(f"Mapped cell {current_cell} with open directions: {sorted(list(open_directions))}")

        else:
            print(f"\nAt cell: {current_cell}. (cached map)")
            open_directions = maze_map[current_cell]

        # ----- Choose next (unvisited) -----
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
            # หมุนไปตามทิศเป้าหมาย: คำนวณมุมหมุนสัมพัทธ์ (difference of dirs * 90)
            turn_angle = (next_move_direction - robot_orientation) * 90
            if turn_angle > 180: turn_angle -= 360
            if turn_angle < -180: turn_angle += 360
            ctrl.turn(turn_angle)

            # เดินหนึ่งช่องด้วย PID ระยะ
            ctrl.move_forward_pid(CELL_SIZE)

            # อัปเดตสถานะการสำรวจ
            visited.add(next_cell)
            path_stack.append(next_cell)
            continue

        # ----- Backtrack -----
        print("Dead end or all neighbors visited. Backtracking...")
        path_stack.pop()

        if not path_stack:
            print("\nReturned to start and all reachable cells explored. Halting.")
            break

        # กลับไปยังเซลล์ก่อนหน้า: หมุน 180 + เดินหนึ่งช่อง
        ctrl.turn(180)
        ctrl.move_forward_pid(CELL_SIZE)

    print("\nDFS exploration complete.")
    print("Displaying final map...")
    mzplot.finalize_show()


if __name__ == "__main__":
    ctrl = Control(conn_type="ap")
    time.sleep(2)
    try:
        dfs_maze_solver(ctrl)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up and closing connection.")
        ctrl.close()
