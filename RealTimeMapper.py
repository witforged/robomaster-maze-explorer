import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk
import threading
import time
import json
import socket

class RobomasterMapper:
    def __init__(self):
        # Map constants
        self.MAP_SIZE = 7
        self.UNKNOWN = 0
        self.FREE = 1
        self.WALL = -1
        self.ROBOT = 2
        
        # Initialize unknown map
        self.map = np.full((7, 7), self.UNKNOWN, dtype=int)
        self.robot_pos = [3, 3]  # Starting position
        self.path_history = []
        self.is_running = False
        
        # Colors for visualization
        self.colors = {
            self.UNKNOWN: '#95a5a6',    # Gray - unknown
            self.FREE: '#ecf0f1',       # Light gray - free space
            self.WALL: '#34495e',       # Dark gray - wall
            self.ROBOT: '#e74c3c'       # Red - robot position
        }
    
    def update_robot_position(self, x, y):
        """Update robot position from external data"""
        if 0 <= x < self.MAP_SIZE and 0 <= y < self.MAP_SIZE:
            self.robot_pos = [x, y]
            self.path_history.append([x, y])
            # Mark current position as free
            self.map[y][x] = self.FREE
            return True
        return False
    
    def update_map_cell(self, x, y, cell_type):
        """Update map cell from sensor data"""
        if 0 <= x < self.MAP_SIZE and 0 <= y < self.MAP_SIZE:
            self.map[y][x] = cell_type
    
    def get_map_for_display(self):
        """Get map with robot position for display"""
        display_map = self.map.copy()
        if self.robot_pos:
            display_map[self.robot_pos[1]][self.robot_pos[0]] = self.ROBOT
        return display_map

class RealTimeGUI:
    def __init__(self):
        self.mapper = RobomasterMapper()
        self.data_receiver = None
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Robomaster Real-Time Mapper")
        self.root.geometry("900x700")
        
        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Control panel
        self.setup_control_panel()
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=100, blit=False
        )
        
        self.update_plot(0)
    
    def setup_control_panel(self):
        """Setup control panel"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Title
        ttk.Label(control_frame, text="Robot Control", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Status: Waiting")
        self.status_label.pack(pady=5)
        
        # Position info
        self.pos_label = ttk.Label(control_frame, text="Position: (3, 3)")
        self.pos_label.pack(pady=5)
        
        # Path length
        self.path_label = ttk.Label(control_frame, text="Path Length: 0")
        self.path_label.pack(pady=5)
        
        # Buttons
        ttk.Button(control_frame, text="Start Listening", 
                  command=self.start_listening).pack(pady=5)
        ttk.Button(control_frame, text="Stop", 
                  command=self.stop_listening).pack(pady=5)
        ttk.Button(control_frame, text="Reset Map", 
                  command=self.reset_map).pack(pady=5)
        ttk.Button(control_frame, text="Save Path", 
                  command=self.save_path).pack(pady=5)
        
        # Manual input section
        ttk.Label(control_frame, text="Manual Input:", 
                 font=('Arial', 12, 'bold')).pack(pady=(20, 5))
        
        # Position input
        pos_frame = ttk.Frame(control_frame)
        pos_frame.pack(pady=5)
        ttk.Label(pos_frame, text="X:").pack(side=tk.LEFT)
        self.x_entry = ttk.Entry(pos_frame, width=5)
        self.x_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(pos_frame, text="Y:").pack(side=tk.LEFT)
        self.y_entry = ttk.Entry(pos_frame, width=5)
        self.y_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_frame, text="Update Position", 
                  command=self.manual_update_position).pack(pady=5)
        
        # Map cell update
        cell_frame = ttk.Frame(control_frame)
        cell_frame.pack(pady=5)
        ttk.Label(cell_frame, text="Cell Type:").pack()
        self.cell_var = tk.StringVar(value="Free")
        ttk.Radiobutton(cell_frame, text="Free", variable=self.cell_var, 
                       value="Free").pack()
        ttk.Radiobutton(cell_frame, text="Wall", variable=self.cell_var, 
                       value="Wall").pack()
        
        ttk.Button(control_frame, text="Update Cell", 
                  command=self.manual_update_cell).pack(pady=5)
    
    def update_plot(self, frame):
        """Update plot with current map and robot position"""
        self.ax.clear()
        
        # Get display map
        display_map = self.mapper.get_map_for_display()
        
        # Create color map
        color_map = np.zeros((7, 7, 3))
        for y in range(7):
            for x in range(7):
                cell_type = display_map[y][x]
                color_hex = self.mapper.colors[cell_type]
                # Convert hex to RGB
                color_rgb = [int(color_hex[i:i+2], 16)/255.0 for i in (1, 3, 5)]
                color_map[y][x] = color_rgb
        
        # Display map
        self.ax.imshow(color_map, origin='upper')
        
        # Draw grid
        for i in range(8):
            self.ax.axhline(i-0.5, color='black', linewidth=1)
            self.ax.axvline(i-0.5, color='black', linewidth=1)
        
        # Draw path
        if len(self.mapper.path_history) > 1:
            path_x = [pos[0] for pos in self.mapper.path_history]
            path_y = [pos[1] for pos in self.mapper.path_history]
            self.ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8)
            
            # Draw path points
            self.ax.scatter(path_x, path_y, c='blue', s=30, alpha=0.6)
        
        # Highlight current robot position
        if self.mapper.robot_pos:
            self.ax.scatter(self.mapper.robot_pos[0], self.mapper.robot_pos[1], 
                          c='red', s=200, marker='o', edgecolors='white', linewidth=2)
        
        self.ax.set_xlim(-0.5, 6.5)
        self.ax.set_ylim(-0.5, 6.5)
        self.ax.set_title('Maze Mapping', fontsize=14)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        # Update status labels
        self.pos_label.config(text=f"Position: ({self.mapper.robot_pos[0]}, {self.mapper.robot_pos[1]})")
        self.path_label.config(text=f"Path Length: {len(self.mapper.path_history)}")
        
        return []
    
    def start_listening(self):
        """Start listening for robot data"""
        self.mapper.is_running = True
        self.status_label.config(text="Status: Listening")
        
        # Start data receiver thread (placeholder for actual implementation)
        self.data_receiver = threading.Thread(target=self.data_receiver_thread)
        self.data_receiver.daemon = True
        self.data_receiver.start()
    
    def stop_listening(self):
        """Stop listening for robot data"""
        self.mapper.is_running = False
        self.status_label.config(text="Status: Stopped")
    
    def reset_map(self):
        """Reset map and path"""
        self.mapper.map = np.full((7, 7), self.mapper.UNKNOWN, dtype=int)
        self.mapper.robot_pos = [3, 3]
        self.mapper.path_history = []
        self.status_label.config(text="Status: Reset")
    
    def save_path(self):
        """Save current path to file"""
        data = {
            'path_history': self.mapper.path_history,
            'map': self.mapper.map.tolist(),
            'robot_position': self.mapper.robot_pos
        }
        
        filename = f"robot_path_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.status_label.config(text=f"Saved: {filename}")
    
    def manual_update_position(self):
        """Manually update robot position"""
        try:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            if self.mapper.update_robot_position(x, y):
                self.status_label.config(text=f"Updated position: ({x}, {y})")
            else:
                self.status_label.config(text="Invalid position!")
        except ValueError:
            self.status_label.config(text="Invalid input!")
    
    def manual_update_cell(self):
        """Manually update map cell"""
        try:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            cell_type = self.mapper.FREE if self.cell_var.get() == "Free" else self.mapper.WALL
            self.mapper.update_map_cell(x, y, cell_type)
            self.status_label.config(text=f"Updated cell ({x}, {y}): {self.cell_var.get()}")
        except ValueError:
            self.status_label.config(text="Invalid input!")
    
    def data_receiver_thread(self):
        """Thread to receive data from external source"""
        # This is a placeholder for actual data receiving logic
        # You can implement socket communication, file reading, or other methods here
        
        while self.mapper.is_running:
            # Example: simulate receiving data
            # In real implementation, replace this with actual data receiving logic
            time.sleep(1)
            
            # Example of how to update robot position from external data:
            # self.mapper.update_robot_position(new_x, new_y)
            # self.mapper.update_map_cell(x, y, cell_type)
            
        print("Data receiver thread stopped")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

# Example of how to integrate with external robot data
class RobotDataInterface:
    """Interface for receiving robot data from external sources"""
    
    def __init__(self, mapper):
        self.mapper = mapper
    
    def update_from_socket(self, host='localhost', port=8888):
        """Receive data via socket"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            
            while self.mapper.is_running:
                data = sock.recv(1024).decode()
                if data:
                    # Parse received data (example format: "POS:x,y" or "CELL:x,y,type")
                    self.parse_and_update(data)
                    
        except Exception as e:
            print(f"Socket error: {e}")
        finally:
            sock.close()
    
    def parse_and_update(self, data):
        """Parse received data and update mapper"""
        try:
            if data.startswith("POS:"):
                # Position update: "POS:3,4"
                coords = data[4:].split(',')
                x, y = int(coords[0]), int(coords[1])
                self.mapper.update_robot_position(x, y)
                
            elif data.startswith("CELL:"):
                # Cell update: "CELL:2,3,1" (x,y,type)
                parts = data[5:].split(',')
                x, y, cell_type = int(parts[0]), int(parts[1]), int(parts[2])
                self.mapper.update_map_cell(x, y, cell_type)
                
        except Exception as e:
            print(f"Parse error: {e}")

if __name__ == "__main__":
    app = RealTimeGUI()
    app.run()
