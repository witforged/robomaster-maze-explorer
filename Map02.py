import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import tkinter as tk
from tkinter import ttk, filedialog

class SimpleMazeMapper:
    def __init__(self):
        self.map = np.zeros((7, 7), dtype=int)
        self.robot_pos = [3, 3]
        self.visited = set()
        self.path = []
        self.exploring = False
        self.generate_maze()
    
    def generate_maze(self):
        """Generate random maze"""
        for y in range(7):
            for x in range(7):
                if x == 3 and y == 3:
                    self.map[y][x] = 1  # Free
                else:
                    self.map[y][x] = -1 if random.random() < 0.3 else 1
    
    def load_csv(self, filename):
        """Load map from CSV"""
        try:
            with open(filename, 'r') as f:
                data = [[int(cell) for cell in row] for row in csv.reader(f)]
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
        
        self.exploring = False
        return False
    
    def get_coverage(self):
        """Get coverage percentage"""
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
        self.mapper.generate_maze()
        self.update_display()
    
    def load_csv(self):
        """Load map from CSV"""
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
                
    def is_valid_position(self, x, y):
        """ตรวจสอบว่าตำแหน่งอยู่ในขอบเขตแผนที่หรือไม่"""
        return 0 <= x < self.MAP_SIZE and 0 <= y < self.MAP_SIZE
    
    def has_visited_or_sensed(self, x, y):
        """ตรวจสอบว่าหุ่นเคยไปหรือสามารถรับรู้ตำแหน่งนี้หรือไม่"""
        # หุ่นสามารถรับรู้เซลล์ข้างเคียงได้
        dist = abs(x - self.robot_position[0]) + abs(y - self.robot_position[1])
        return dist <= 1 or f"{x},{y}" in self.visited_cells
    
    def is_frontier_cell(self, x, y):
        """ตรวจสอบว่าเป็น frontier cell หรือไม่"""
        if self.has_visited_or_sensed(x, y) or self.map[y][x] == self.WALL:
            return False
            
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            adj_x, adj_y = x + dx, y + dy
            if (self.is_valid_position(adj_x, adj_y) and 
                self.has_visited_or_sensed(adj_x, adj_y)):
                return True
        return False
    
    def update_display_map(self):
        """อัปเดตแผนที่สำหรับแสดงผล"""
        self.display_map = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=int)
        
        for y in range(self.MAP_SIZE):
            for x in range(self.MAP_SIZE):
                # ตรวจสอบสถานะของแต่ละเซลล์
                if not self.has_visited_or_sensed(x, y):
                    self.display_map[y][x] = self.UNKNOWN
                elif self.map[y][x] == self.WALL:
                    self.display_map[y][x] = self.WALL
                elif f"{x},{y}" in self.visited_cells:
                    self.display_map[y][x] = self.VISITED
                else:
                    self.display_map[y][x] = self.FREE
                
                # ตำแหน่งปัจจุบัน
                if x == self.robot_position[0] and y == self.robot_position[1]:
                    self.display_map[y][x] = self.CURRENT
                
                # Frontier cells
                elif self.is_frontier_cell(x, y):
                    self.display_map[y][x] = self.FRONTIER
    
    def calculate_frontier_score(self, x, y):
        """คำนวณคะแนน frontier สำหรับการตัดสินใจ"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        score = 0
        
        for dx, dy in directions:
            adj_x, adj_y = x + dx, y + dy
            if (self.is_valid_position(adj_x, adj_y) and 
                not self.has_visited_or_sensed(adj_x, adj_y)):
                score += 1
                
        return score
    
    def find_best_move(self):
        """หาการเคลื่อนที่ที่ดีที่สุด"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        possible_moves = []
        
        for dx, dy in directions:
            new_x = self.robot_position[0] + dx
            new_y = self.robot_position[1] + dy
            
            if (self.is_valid_position(new_x, new_y) and 
                self.map[new_y][new_x] != self.WALL):
                
                is_visited = f"{new_x},{new_y}" in self.visited_cells
                frontier_score = self.calculate_frontier_score(new_x, new_y)
                priority = 0 if is_visited else frontier_score + 10
                
                possible_moves.append({
                    'position': [new_x, new_y],
                    'is_visited': is_visited,
                    'frontier_score': frontier_score,
                    'priority': priority
                })
        
        if not possible_moves:
            return None
            
        # เรียงตามลำดับความสำคัญ (เซลล์ที่ยังไม่เคยเดินและมี frontier score สูง)
        possible_moves.sort(key=lambda x: x['priority'], reverse=True)
        return possible_moves[0]['position']
    
    def explore_step(self):
        """ขั้นตอนการสำรวจหนึ่งก้าว"""
        # บันทึกตำแหน่งปัจจุบันเป็น visited
        current_key = f"{self.robot_position[0]},{self.robot_position[1]}"
        self.visited_cells.add(current_key)
        self.path_history.append(self.robot_position.copy())
        
        # หาการเคลื่อนที่ที่ดีที่สุด
        next_move = self.find_best_move()
        
        if next_move:
            self.robot_position = next_move
            self.step_count += 1
            self.total_steps += 1
            return True
        else:
            # ไม่สามารถเคลื่อนที่ต่อได้
            self.exploration_complete = True
            return False
    
    def is_exploration_complete(self):
        """ตรวจสอบว่าการสำรวจเสร็จสิ้นหรือไม่"""
        total_reachable = np.sum(self.map != self.WALL)
        return len(self.visited_cells) >= total_reachable * 0.95  # 95% completion
    
    def get_stats(self):
        """คำนวณสถิติต่างๆ"""
        total_cells = self.MAP_SIZE * self.MAP_SIZE
        explored_cells = len(self.visited_cells)
        coverage = (explored_cells / total_cells) * 100
        
        elapsed_time = 0
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            
        efficiency = (explored_cells / elapsed_time * 10) if elapsed_time > 0 else 0
        redundancy = ((self.total_steps - explored_cells) / self.total_steps * 100) if self.total_steps > 0 else 0
        
        return {
            'coverage': coverage,
            'explored_cells': explored_cells,
            'total_cells': total_cells,
            'elapsed_time': elapsed_time,
            'efficiency': efficiency,
            'step_count': self.step_count,
            'redundancy': redundancy
        }
    
    def export_map_data(self, filename=None):
        """ส่งออกข้อมูลแผนที่"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"maze_map_{timestamp}.json"
            
        stats = self.get_stats()
        
        map_data = {
            'timestamp': datetime.now().isoformat(),
            'map_size': self.MAP_SIZE,
            'true_map': self.map.tolist(),
            'display_map': self.display_map.tolist(),
            'robot_position': self.robot_position,
            'visited_cells': list(self.visited_cells),
            'path_history': self.path_history,
            'stats': stats
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(map_data, f, ensure_ascii=False, indent=2)
            
        return filename

class MazeGUI:
    def __init__(self):
        self.system = MazeMappingSystem()
        self.exploration_thread = None
        
        # สร้าง main window
        self.root = tk.Tk()
        self.root.title("Robomaster Maze Mapping")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        self.setup_gui()
        self.update_display()
        
        # เริ่มการสำรวจอัตโนมัติ
        self.start_exploration()
        
    def setup_gui(self):
        """ตั้งค่า GUI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Robomaster Maze Mapping", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left frame for map
        self.map_frame = ttk.LabelFrame(main_frame, text="Maze Map 7×7", padding="10")
        self.map_frame.grid(row=1, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right frame for controls and stats
        control_frame = ttk.LabelFrame(main_frame, text="การควบคุมและสถิติ", padding="10")
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Setup matplotlib figure for map
        self.setup_map_display()
        

        # Setup stats display
        self.setup_stats_display(control_frame)
        
        # Setup log display
        self.setup_log_display(control_frame)
        
    def setup_map_display(self):
        """ตั้งค่าการแสดงผลแผนที่"""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.map_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Legend frame
        legend_frame = ttk.Frame(self.map_frame)
        legend_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create legend
        for i, (state, color) in enumerate(self.system.colors.items()):
            if state == self.system.CURRENT:  # Skip current position in legend
                continue
            legend_label = ttk.Label(legend_frame, text=f"● {self.system.color_labels[state]}")
            legend_label.grid(row=i//2, column=i%2, padx=10, pady=2, sticky=tk.W)
            


    def setup_stats_display(self, parent):
        """ตั้งค่าการแสดงสถิติ"""
        stats_frame = ttk.LabelFrame(parent, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Stats labels
        self.stats_labels = {}
        stats_items = [
            ('coverage', 'Coverage'),
            ('cells', 'Cells'),
            ('time', 'Time'),
            ('efficiency', 'Efficiency'),
            ('steps', 'Steps'),
            ('redundancy', 'Redundancy')
        ]
        
        for i, (key, label) in enumerate(stats_items):
            frame = ttk.Frame(stats_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            self.stats_labels[key] = ttk.Label(frame, text="0", font=('Arial', 10, 'bold'))
            self.stats_labels[key].pack(side=tk.RIGHT)
            
        # Progress bar
        self.progress_var = tk.StringVar()
        self.progress = ttk.Progressbar(stats_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(10, 5))
        
        self.progress_label = ttk.Label(stats_frame, textvariable=self.progress_var)
        self.progress_label.pack()
        
    def setup_log_display(self, parent):
        """ตั้งค่าการแสดง log"""
        log_frame = ttk.LabelFrame(parent, text="Activity Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text widget with scrollbar
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(text_frame, height=10, width=40, font=('Courier', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.add_log("System ready - click start to begin")
        
    def update_display(self):
        """อัปเดตการแสดงผลทั้งหมด"""
        self.system.update_display_map()
        self.update_map()
        self.update_stats()
        
    def update_map(self):
        """อัปเดตแผนที่"""
        self.ax.clear()
        
        # สร้างแผนที่สี
        color_map = np.zeros((self.system.MAP_SIZE, self.system.MAP_SIZE, 3))
        
        for y in range(self.system.MAP_SIZE):
            for x in range(self.system.MAP_SIZE):
                state = self.system.display_map[y][x]
                color_hex = self.system.colors[state]
                # แปลง hex เป็น RGB
                color_rgb = [int(color_hex[i:i+2], 16)/255.0 for i in (1, 3, 5)]
                color_map[y][x] = color_rgb
        
        self.ax.imshow(color_map, origin='upper')
        
        # วาดเส้นทางที่หุ่นยนต์เดิน
        if len(self.system.path_history) > 1:
            path_x = [pos[0] for pos in self.system.path_history]
            path_y = [pos[1] for pos in self.system.path_history]
            
            # วาดเส้นทางเป็นเส้นสีน้ำเงินหนา
            self.ax.plot(path_x, path_y, color='#3498db', linewidth=4, alpha=0.9, 
                        marker='o', markersize=6, markerfacecolor='#2980b9', 
                        markeredgecolor='white', markeredgewidth=2, label='Way Robot')
            
            # เพิ่มลูกศรแสดงทิศทาง
            for i in range(len(path_x) - 1):
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                if dx != 0 or dy != 0:  # ตรวจสอบว่าไม่ใช่จุดเดียวกัน
                    self.ax.annotate('', xy=(path_x[i+1], path_y[i+1]), 
                                   xytext=(path_x[i], path_y[i]),
                                   arrowprops=dict(arrowstyle='->', color='#2980b9', 
                                                 lw=3, alpha=0.8))
        

        
        # เพิ่มป้ายกำกับพิกัดพร้อมเอฟเฟกต์สี
        for y in range(self.system.MAP_SIZE):
            for x in range(self.system.MAP_SIZE):
                cell_state = self.system.display_map[y][x]
                
                # เลือกสีข้อความตามสถานะของเซลล์
                if cell_state == self.system.VISITED or cell_state == self.system.CURRENT:
                    text_color = 'darkgreen'
                    font_weight = 'bold'
                else:
                    text_color = 'black'
                    font_weight = 'normal'
                
                self.ax.text(x, y, f'{x},{y}', ha='center', va='center', 
                           fontsize=8, color=text_color, weight=font_weight)
        
        # เพิ่มเส้นกริด
        self.ax.set_xticks(np.arange(-0.5, self.system.MAP_SIZE, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.system.MAP_SIZE, 1), minor=True)
        self.ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        
        self.ax.set_title("Maze Map", fontsize=14, fontweight='bold')
        self.ax.set_xlim(-0.5, self.system.MAP_SIZE-0.5)
        self.ax.set_ylim(-0.5, self.system.MAP_SIZE-0.5)
        
        # ซ่อน ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # เพิ่ม legend หากมีเส้นทาง
        if len(self.system.path_history) > 1:
            self.ax.legend(loc='upper right', bbox_to_anchor=(1, 1), 
                         framealpha=0.9, fancybox=True, shadow=True)
        
        self.canvas.draw()
        
    def update_stats(self):
        """อัปเดตสถิติ"""
        stats = self.system.get_stats()
        
        self.stats_labels['coverage'].config(text=f"{stats['coverage']:.1f}%")
        self.stats_labels['cells'].config(text=f"{stats['explored_cells']}/{stats['total_cells']}")
        self.stats_labels['time'].config(text=f"{stats['elapsed_time']:.1f}s")
        self.stats_labels['efficiency'].config(text=f"{stats['efficiency']:.1f}%")
        self.stats_labels['steps'].config(text=str(stats['step_count']))
        self.stats_labels['redundancy'].config(text=f"{stats['redundancy']:.1f}%")
        
        # Progress bar
        self.progress['value'] = stats['coverage']
        self.progress_var.set(f"{stats['coverage']:.1f}%")
        
    def add_log(self, message, level="INFO"):
        """เพิ่มข้อความใน log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
    def start_exploration(self):
        """เริ่มการสำรวจ"""
        if not self.system.is_exploring:
            self.system.is_exploring = True
            self.system.is_paused = False
            self.system.start_time = time.time()
            
            self.exploration_thread = threading.Thread(target=self.exploration_loop)
            self.exploration_thread.daemon = True
            self.exploration_thread.start()
            
            self.add_log("เริ่มการสำรวจเขาวงกต")
        elif self.system.is_paused:
            self.system.is_paused = False
            self.add_log("ดำเนินการสำรวจต่อ")
            
    def pause_exploration(self):
        """หยุดการสำรวจชั่วคราว"""
        if self.system.is_exploring:
            self.system.is_paused = not self.system.is_paused
            status = "หยุดชั่วคราว" if self.system.is_paused else "ดำเนินการต่อ"
            self.add_log(f"การสำรวจ: {status}")
            
    def reset_exploration(self):
        """รีเซ็ตการสำรวจ"""
        self.system.is_exploring = False
        self.system.is_paused = False
        self.system.visited_cells.clear()
        self.system.path_history.clear()
        self.system.robot_position = [3, 3]
        self.system.start_time = None
        self.system.step_count = 0
        self.system.total_steps = 0
        self.system.exploration_complete = False
        
        self.update_display()
        self.add_log("รีเซ็ตระบบเรียบร้อย")
        
    def generate_new_maze(self):
        """สร้างเขาวงกตใหม่"""
        self.reset_exploration()
        self.system.generate_random_maze()
        self.update_display()
        self.add_log("สร้างเขาวงกตใหม่เรียบร้อย")
        
    def load_csv_map(self):
        """โหลดแผนที่จากไฟล์ CSV"""
        try:
            filename = filedialog.askopenfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="เลือกไฟล์ CSV แผนที่"
            )
            
            if filename:
                success = self.system.load_map_from_csv(filename)
                if success:
                    self.reset_exploration()
                    self.update_display()
                    self.add_log(f"โหลดแผนที่จาก CSV: {filename}")
                    messagebox.showinfo("สำเร็จ", f"โหลดแผนที่จากไฟล์ CSV เรียบร้อย\n{filename}")
                else:
                    self.add_log(f"ไม่สามารถโหลดไฟล์ CSV: {filename}", "ERROR")
                    messagebox.showerror("ข้อผิดพลาด", "ไม่สามารถโหลดไฟล์ CSV ได้\nกรุณาตรวจสอบรูปแบบไฟล์")
        except Exception as e:
            self.add_log(f"ข้อผิดพลาดในการโหลด CSV: {str(e)}", "ERROR")
            
    def export_map(self):
        """ส่งออกข้อมูลแผนที่"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="บันทึกข้อมูลแผนที่"
            )
            
            if filename:
                self.system.export_map_data(filename)
                self.add_log(f"ส่งออกแผนที่: {filename}")
                messagebox.showinfo("สำเร็จ", f"ส่งออกข้อมูลแผนที่เรียบร้อย\n{filename}")
        except Exception as e:
            self.add_log(f"ข้อผิดพลาดในการส่งออก: {str(e)}", "ERROR")
            
    def exploration_loop(self):
        """วนลูปการสำรวจ"""
        while (self.system.is_exploring and 
               not self.system.exploration_complete):
            
            if not self.system.is_paused:
                success = self.system.explore_step()
                
                # อัปเดต GUI ใน main thread
                self.root.after(0, self.update_display)
                
                if not success:
                    self.system.is_exploring = False
                    self.root.after(0, lambda: self.add_log("การสำรวจเสร็จสิ้น"))
                    break
                    
                if self.system.is_exploration_complete():
                    self.system.is_exploring = False
                    self.root.after(0, lambda: self.add_log("สำรวจเขาวงกตครบ 95% แล้ว!"))
                    break
                    
            time.sleep(0.5)  # หน่วงเวลาเพื่อให้เห็นการเคลื่อนที่
            
    def run(self):
        """เริ่มต้นโปรแกรม"""
        self.root.mainloop()

if __name__ == "__main__":
    # สร้างและรัน GUI
    app = MazeGUI()
    app.run()