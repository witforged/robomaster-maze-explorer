import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import random
import threading
import tkinter as tk
from tkinter import ttk

class SimpleMazeMapper:
    def __init__(self):
        self.MAP_SIZE = 7
        self.map = np.zeros((7, 7), dtype=int)
        self.robot_pos = [3, 3]
        self.visited = set()
        self.path = []
        self.exploring = False
        
        # Generate random maze
        for y in range(7):
            for x in range(7):
                if x == 3 and y == 3:
                    self.map[y][x] = 1  # Free
                else:
                    self.map[y][x] = -1 if random.random() < 0.3 else 1  # Wall or Free
    
    def load_csv(self, filename):
        """Load map from CSV file"""
        try:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                data = [[int(cell) for cell in row] for row in reader]
                if len(data) == 7 and all(len(row) == 7 for row in data):
                    self.map = np.array(data)
                    return True
        except:
            pass
        return False
    
    def explore_step(self):
        """Single exploration step"""
        if not self.exploring:
            return False
            
        x, y = self.robot_pos
        self.visited.add((x, y))
        self.path.append((x, y))
        
        # Find next move
        moves = [(0,1), (1,0), (0,-1), (-1,0)]
        random.shuffle(moves)
        
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if (0 <= nx < 7 and 0 <= ny < 7 and 
                self.map[ny][nx] == 1 and (nx, ny) not in self.visited):
                self.robot_pos = [nx, ny]
                return True
        
        # No valid moves
        self.exploring = False
        return False
    
    def get_coverage(self):
        """Get exploration coverage percentage"""
        free_cells = np.sum(self.map == 1)
        return len(self.visited) / free_cells * 100 if free_cells > 0 else 0

class SimpleGUI:
    def __init__(self):
        self.mapper = SimpleMazeMapper()
        
        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Simple Maze Mapper")
        self.root.geometry("800x600")
        
        # Map display
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Start", command=self.start).pack(pady=5)
        ttk.Button(control_frame, text="Stop", command=self.stop).pack(pady=5)
        ttk.Button(control_frame, text="Reset", command=self.reset).pack(pady=5)
        ttk.Button(control_frame, text="New Maze", command=self.new_maze).pack(pady=5)
        ttk.Button(control_frame, text="Load CSV", command=self.load_csv).pack(pady=5)
        
        # Stats
        self.stats_label = ttk.Label(control_frame, text="Coverage: 0%")
        self.stats_label.pack(pady=10)
        
        self.update_display()
    
    def update_display(self):
        """Update map display"""
        self.ax.clear()
        
        # Draw map
        for y in range(7):
            for x in range(7):
                color = 'black' if self.mapper.map[y][x] == -1 else 'white'
                if (x, y) in self.mapper.visited:
                    color = 'lightgreen'
                if [x, y] == self.mapper.robot_pos:
                    color = 'red'
                
                self.ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, 
                                              facecolor=color, edgecolor='gray'))
        
        # Draw path
        if len(self.mapper.path) > 1:
            path_x, path_y = zip(*self.mapper.path)
            self.ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
        
        self.ax.set_xlim(-0.5, 6.5)
        self.ax.set_ylim(-0.5, 6.5)
        self.ax.set_aspect('equal')
        self.ax.set_title('Maze Exploration')
        
        # Update stats
        coverage = self.mapper.get_coverage()
        self.stats_label.config(text=f"Coverage: {coverage:.1f}%")
        
        self.canvas.draw()
    
    def start(self):
        """Start exploration"""
        self.mapper.exploring = True
        self.explore_loop()
    
    def stop(self):
        """Stop exploration"""
        self.mapper.exploring = False
    
    def reset(self):
        """Reset exploration"""
        self.mapper.exploring = False
        self.mapper.robot_pos = [3, 3]
        self.mapper.visited.clear()
        self.mapper.path.clear()
        self.update_display()
    
    def new_maze(self):
        """Generate new maze"""
        self.reset()
        self.mapper.__init__()
        self.update_display()
    
    def load_csv(self):
        """Load map from CSV"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename and self.mapper.load_csv(filename):
            self.reset()
            self.update_display()
    
    def explore_loop(self):
        """Exploration loop"""
        if self.mapper.exploring:
            success = self.mapper.explore_step()
            self.update_display()
            
            if success:
                self.root.after(500, self.explore_loop)
            else:
                self.mapper.exploring = False
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SimpleGUI()
    app.run()
