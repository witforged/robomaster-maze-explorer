# -*- coding:utf-8 -*-
"""
plot.py
- ฟังก์ชันพล็อตสถานะการสำรวจเขาวงกตแบบเรียลไทม์
"""

import matplotlib.pyplot as plt
import numpy as np

plt.ion()
_fig, _ax = plt.subplots(figsize=(8, 8))

def plot_maze(current_cell, visited, walls, path_stack, title="Real-time Maze Exploration"):
    ax = _ax
    ax.clear()

    # ช่องที่เยือนแล้ว
    for (x, y) in visited:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                   facecolor='lightgray', edgecolor='gray'))

    # ผนัง (คั่นระหว่างสอง cell ที่อยู่ติดกัน)
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        if x1 == x2:  # ผนังแนวนอน
            ax.plot([x1 - 0.5, x1 + 0.5],
                    [max(y1, y2) - 0.5, max(y1, y2) - 0.5],
                    'k-', linewidth=4)
        else:         # ผนังแนวตั้ง
            ax.plot([max(x1, x2) - 0.5, max(x1, x2) - 0.5],
                    [y1 - 0.5, y1 + 0.5],
                    'k-', linewidth=4)

    # เส้นทางที่เดิน
    if len(path_stack) > 1:
        path_x, path_y = zip(*path_stack)
        ax.plot(path_x, path_y, 'b-o', markersize=5, label='Path')

    # ตำแหน่งหุ่น
    cx, cy = current_cell
    ax.plot(cx, cy, 'ro', markersize=12, label='Robot')

    all_x = [c[0] for c in visited]
    all_y = [c[1] for c in visited]
    if not all_x: all_x = [0]
    if not all_y: all_y = [0]

    ax.set_xlim(min(all_x) - 1.5, max(all_x) + 1.5)
    ax.set_ylim(min(all_y) - 1.5, max(all_y) + 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(min(all_x) - 2, max(all_x) + 3, 1))
    ax.set_yticks(np.arange(min(all_y) - 2, max(all_y) + 3, 1))
    ax.grid(True)
    ax.set_title(title)
    plt.pause(0.1)

def finalize_show():
    plt.ioff()
    try:
        plt.show()
    except Exception:
        pass
