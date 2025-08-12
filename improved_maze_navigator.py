import robomaster
from robomaster import robot
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# --- Global Variables ---
last_distance = None      # ระยะกำแพง
current_x = 0             # ตำแหน่งหุ่นแกน x (จาก Position Subscription)
current_y = 0             # ตำแหน่งหุ่นแกน y (จาก Position Subscription)
current_yaw = 0.0         # มุมหุ่น

# --- Enhanced Maze and Mapping Variables ---
maze_map = {}             # เก็บข้อมูลแผนที่ {(x,y): 'free'/'wall'/'unknown'}
visited = set([(0, 0)])   # เซลล์ที่เยี่ยมชมแล้ว
path_stack = [(0, 0)]     # Stack สำหรับ backtracking
walls = set()             # ตำแหน่งกำแพงที่ตรวจพบ
robot_path = [(0, 0)]     # เส้นทางจริงที่หุ่นเดิน
current_grid_pos = (0, 0) # ตำแหน่งปัจจุบันใน grid

# --- Constants ---
WALL_THRESHOLD = 60   # ระยะน้อยกว่า 60 มีกำแพง
CELL_SIZE = 0.6       # ขนาด grid (เมตร)
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # North, East, South, West

# --- Plotting Setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(10, 10))

# --- Enhanced PID Controller Class ---
class PIDController:
    """Enhanced PID controller class with better tuning."""
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        
    def update(self, current_value, dt=0.1):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output
    
    def reset(self):
        self.previous_error = 0
        self.integral = 0

# --- Enhanced Navigation Functions ---
class MazeNavigator:
    def __init__(self):
        self.current_grid_x = 0
        self.current_grid_y = 0
        self.visited_cells = set([(0, 0)])
        self.exploration_stack = [(0, 0)]
        self.walls_detected = set()
        self.real_path = [(0, 0)]
        
    def get_discretized_orientation(self, yaw):
        """แปลงมุมหมุนเป็นทิศทางแบบดิจิทัล"""
        if -45 <= yaw < 45: 
            return 0  # North
        elif 45 <= yaw < 135: 
            return 1  # East
        elif -135 > yaw >= -180 or 135 <= yaw <= 180: 
            return 2  # South
        elif -135 <= yaw < -45: 
            return 3  # West
        return 0
    
    def get_target_coordinates(self, grid_x, grid_y, direction):
        """คำนวณพิกัดเป้าหมายจากทิศทาง"""
        dx, dy = DIRECTIONS[direction]
        return grid_x + dx, grid_y + dy
    
    def get_relative_directions(self, robot_orientation):
        """คำนวณทิศทางสัมพัทธ์"""
        front = robot_orientation
        right = (robot_orientation + 1) % 4
        left = (robot_orientation - 1 + 4) % 4
        back = (robot_orientation + 2) % 4
        return {"F": front, "R": right, "L": left, "B": back}
    
    def scan_surroundings(self, ep_sensor):
        """สแกนสิ่งแวดล้อมและอัปเดตแผนที่"""
        current_orientation = self.get_discretized_orientation(current_yaw)
        directions = self.get_relative_directions(current_orientation)
        
        detected_walls = {}
        
        # สแกนทิศทางต่างๆ
        for direction_name, direction_index in directions.items():
            # หมุนไปยังทิศทางที่ต้องการสแกน
            target_yaw = direction_index * 90
            self.rotate_to_angle(target_yaw)
            
            # วัดระยะ
            distance = ep_sensor.get_distance_sensor()
            if distance is not None and distance < WALL_THRESHOLD:
                # มีกำแพง
                target_x, target_y = self.get_target_coordinates(
                    self.current_grid_x, self.current_grid_y, direction_index
                )
                self.walls_detected.add((target_x, target_y))
                detected_walls[direction_index] = True
            else:
                detected_walls[direction_index] = False
        
        return detected_walls
    
    def find_unvisited_neighbors(self):
        """หาเซลล์ข้างเคียงที่ยังไม่เยี่ยมชม"""
        unvisited = []
        for i, (dx, dy) in enumerate(DIRECTIONS):
            neighbor_x = self.current_grid_x + dx
            neighbor_y = self.current_grid_y + dy
            neighbor_pos = (neighbor_x, neighbor_y)
            
            # ตรวจสอบว่าไม่ใช่กำแพงและยังไม่เยี่ยมชม
            if (neighbor_pos not in self.walls_detected and 
                neighbor_pos not in self.visited_cells):
                unvisited.append((neighbor_x, neighbor_y, i))
        
        return unvisited
    
    def backtrack_to_unvisited(self):
        """Backtrack กลับไปยังเซลล์ที่มีทางที่ยังไม่เยี่ยมชม"""
        while self.exploration_stack:
            # ย้อนกลับไปยังเซลล์ก่อนหน้า
            prev_x, prev_y = self.exploration_stack.pop()
            
            # อัปเดตตำแหน่งปัจจุบัน
            self.current_grid_x = prev_x
            self.current_grid_y = prev_y
            
            # เพิ่มในเส้นทางจริง
            self.real_path.append((prev_x, prev_y))
            
            # ตรวจสอบว่ามีทางที่ยังไม่เยี่ยมชมหรือไม่
            unvisited = self.find_unvisited_neighbors()
            if unvisited:
                # เลือกทิศทางแรกที่หาได้
                target_x, target_y, direction = unvisited[0]
                return target_x, target_y, direction
        
        return None, None, None
    
    def choose_next_move(self, detected_walls):
        """เลือกการเคลื่อนที่ครั้งต่อไป"""
        # หาเซลล์ข้างเคียงที่ยังไม่เยี่ยมชม
        unvisited = self.find_unvisited_neighbors()
        
        if unvisited:
            # มีทางที่ยังไม่เยี่ยมชม - เลือกทางแรก
            target_x, target_y, direction = unvisited[0]
            return target_x, target_y, direction
        else:
            # ทางตัน - ต้อง backtrack
            print("Dead end detected! Backtracking...")
            return self.backtrack_to_unvisited()
    
    def move_to_target(self, target_x, target_y, ep_chassis):
        """เคลื่อนที่ไปยังเป้าหมาย"""
        # คำนวณระยะทางที่ต้องเคลื่อนที่
        dx = target_x - self.current_grid_x
        dy = target_y - self.current_grid_y
        
        # เคลื่อนที่ตามแกน
        if dx != 0:
            ep_chassis.move(x=dx * CELL_SIZE, y=0, z=0, xy_speed=0.3).wait_for_completed()
        if dy != 0:
            ep_chassis.move(x=0, y=dy * CELL_SIZE, z=0, xy_speed=0.3).wait_for_completed()
        
        # อัปเดตตำแหน่ง
        self.current_grid_x = target_x
        self.current_grid_y = target_y
        self.visited_cells.add((target_x, target_y))
        self.exploration_stack.append((target_x, target_y))
        self.real_path.append((target_x, target_y))
        
        print(f"Moved to grid position: ({target_x}, {target_y})")
    
    def rotate_to_angle(self, target_angle):
        """หมุนไปยังมุมเป้าหมาย"""
        # คำนวณมุมที่ต้องหมุน
        angle_diff = target_angle - current_yaw
        
        # ปรับให้อยู่ในช่วง -180 ถึง 180
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        if abs(angle_diff) > 5:  # หมุนถ้าต่างกันมากกว่า 5 องศา
            ep_chassis.move(x=0, y=0, z=angle_diff, z_speed=45).wait_for_completed()

# --- Enhanced Plotting Function ---
def plot_enhanced_maze(navigator):
    """แสดงแผนที่ที่ปรับปรุงแล้ว"""
    ax.clear()
    
    # แสดงเซลล์ที่เยี่ยมชมแล้ว
    for (x, y) in navigator.visited_cells:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                 facecolor='lightblue', edgecolor='blue', alpha=0.7))
    
    # แสดงกำแพงที่ตรวจพบ
    for (x, y) in navigator.walls_detected:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                 facecolor='red', edgecolor='darkred', alpha=0.8))
    
    # แสดงเส้นทางจริงที่หุ่นเดิน
    if len(navigator.real_path) > 1:
        path_x = [pos[0] for pos in navigator.real_path]
        path_y = [pos[1] for pos in navigator.real_path]
        ax.plot(path_x, path_y, 'g-', linewidth=3, alpha=0.8, label='Robot Path')
    
    # แสดงตำแหน่งปัจจุบันของหุ่น
    ax.add_patch(plt.Circle((navigator.current_grid_x, navigator.current_grid_y), 
                          0.3, facecolor='orange', edgecolor='darkorange'))
    
    # แสดงตำแหน่งเริ่มต้น
    ax.add_patch(plt.Circle((0, 0), 0.2, facecolor='green', edgecolor='darkgreen'))
    
    # ตั้งค่าแกนและ grid
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (Grid Units)')
    ax.set_ylabel('Y (Grid Units)')
    ax.set_title('Enhanced Maze Navigation with Real Path Tracking')
    ax.legend()
    
    plt.draw()
    plt.pause(0.1)

# --- Sensor Callback Functions ---
def sub_position_handler(position_info):
    """รับข้อมูลตำแหน่งจาก sensor"""
    global current_x, current_y, current_yaw
    current_x, current_y, current_yaw = position_info

def sub_distance_handler(distance_info):
    """รับข้อมูลระยะทางจาก sensor"""
    global last_distance
    last_distance = distance_info[0]

# --- Main Navigation Function ---
def enhanced_maze_exploration(ep_robot, ep_chassis, ep_sensor):
    """ฟังก์ชันหลักสำหรับการสำรวจเขาวงกตที่ปรับปรุงแล้ว"""
    navigator = MazeNavigator()
    
    # เริ่ม subscription
    ep_robot.sub_position(freq=10, callback=sub_position_handler)
    ep_sensor.sub_distance(freq=10, callback=sub_distance_handler)
    
    try:
        exploration_count = 0
        max_explorations = 100  # จำกัดจำนวนการสำรวจ
        
        while exploration_count < max_explorations:
            print(f"\n--- Exploration Step {exploration_count + 1} ---")
            print(f"Current position: ({navigator.current_grid_x}, {navigator.current_grid_y})")
            
            # สแกนสิ่งแวดล้อม
            detected_walls = navigator.scan_surroundings(ep_sensor)
            print(f"Detected walls: {detected_walls}")
            
            # เลือกการเคลื่อนที่ครั้งต่อไป
            target_x, target_y, direction = navigator.choose_next_move(detected_walls)
            
            if target_x is None:
                print("Exploration complete! No more unvisited areas.")
                break
            
            print(f"Moving to: ({target_x}, {target_y})")
            
            # เคลื่อนที่ไปยังเป้าหมาย
            navigator.move_to_target(target_x, target_y, ep_chassis)
            
            # อัปเดตการแสดงผล
            plot_enhanced_maze(navigator)
            
            exploration_count += 1
            time.sleep(0.5)  # หน่วงเวลาเล็กน้อย
        
        print(f"\nExploration Summary:")
        print(f"Visited cells: {len(navigator.visited_cells)}")
        print(f"Walls detected: {len(navigator.walls_detected)}")
        print(f"Path length: {len(navigator.real_path)}")
        
    finally:
        # หยุด subscription
        ep_robot.unsub_position()
        ep_sensor.unsub_distance()
        
        # หยุดหุ่น
        ep_chassis.drive_speed(x=0, y=0, z=0)
        print("Robot stopped and subscriptions ended.")

# --- Main Execution ---
if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_chassis = ep_robot.chassis
    ep_sensor = ep_robot.sensor
    
    try:
        print("Starting Enhanced Maze Exploration...")
        enhanced_maze_exploration(ep_robot, ep_chassis, ep_sensor)
        
        # รอให้ผู้ใช้ดูผลลัพธ์
        input("Press Enter to close the program...")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        ep_robot.close()
        plt.close()
