# -*- coding:utf-8 -*-
"""
Control.py
- จัดการเชื่อมต่อ RoboMaster + callbacks (position/yaw/distance)
- คำสั่งพื้นฐาน: read_distance_at(), eye(), turn(), move_forward_pid()
- เปิดเป็นคลาส Control ให้ planing.py เรียกใช้งาน
"""

import time
import math
import numpy as np
from robomaster import robot

class PIDController:
    """PID สำหรับคุม 'ระยะที่เดิน' ให้ถึงเป้าอย่างแม่นยำ"""
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self._prev_error = 0.0
        self._integral = 0.0
        self._last_time = time.time()

    def compute(self, current_value):
        t = time.time()
        dt = t - self._last_time
        if dt <= 0:
            return 0.0

        error = self.setpoint - current_value
        P_out = self.Kp * error
        self._integral += error * dt
        I_out = self.Ki * self._integral
        derivative = (error - self._prev_error) / dt
        D_out = self.Kd * derivative

        out = P_out + I_out + D_out
        self._prev_error = error
        self._last_time = t
        return out


class Control:
    """
    ครอบ SDK ของ RoboMaster:
      - subscribe: attitude(yaw), position(x,y), distance(ToF หน้ากิมบอล)
      - การสแกนกำแพงด้วยกิมบอล (F/L/R)
      - หมุนตัวหุ่น turn(angle)
      - เดิน 1 ช่องด้วย PID ระยะจาก odometry (move_forward_pid)
    """
    def __init__(self, conn_type="ap"):
        # ---- Connect robot ----
        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type=conn_type)

        self.ep_chassis = self.ep_robot.chassis
        self.ep_gimbal  = self.ep_robot.gimbal
        self.ep_sensor  = self.ep_robot.sensor

        # ---- States (updated by callbacks) ----
        self.last_distance_cm = None   # cm (จาก ToF ด้านหน้ากิมบอล)
        self.current_x = 0.0           # m
        self.current_y = 0.0           # m
        self.current_yaw = 0.0         # deg

        # ---- Subscribe handlers ----
        def _dist_cb(sub_info):
            # บางรุ่นส่ง list; เอา index 0 เป็นด้านหน้า
            try:
                mm = int(sub_info[0]) if isinstance(sub_info, (list, tuple)) else int(sub_info)
                if mm > 0:
                    self.last_distance_cm = mm / 10.0
            except Exception:
                pass

        def _att_cb(attitude_info):
            # โค้ดต้นฉบับ: yaw, pitch, roll = attitude_info
            try:
                yaw, pitch, roll = attitude_info
                self.current_yaw = float(yaw)
            except Exception:
                pass

        def _pos_cb(position_info):
            # position_info = (x, y, z) หน่วยเป็นเมตร
            try:
                x, y, z = position_info
                self.current_x = float(x)
                self.current_y = float(y)
            except Exception:
                pass

        # ---- Center gimbal & start subscriptions ----
        try:
            self.ep_gimbal.recenter().wait_for_completed()
        except Exception:
            pass

        self.ep_chassis.sub_attitude(freq=10, callback=_att_cb)
        self.ep_chassis.sub_position(freq=50, callback=_pos_cb)
        # อย่าสมัคร distance ค้างยาว ๆ (อ่านทีละช่วงขณะสแกน)
        self._dist_subscribed = False
        self._dist_cb = _dist_cb

        time.sleep(1.0)  # ให้ค่าคงที่ก่อนเริ่มงาน

    # -------- Accessors --------
    def get_yaw_deg(self):
        return self.current_yaw

    def get_xy_m(self):
        return self.current_x, self.current_y

    # -------- Distance read helpers --------
    def _sub_distance(self, freq=20):
        if not self._dist_subscribed:
            self.ep_sensor.sub_distance(freq=freq, callback=self._dist_cb)
            self._dist_subscribed = True

    def _unsub_distance(self):
        if self._dist_subscribed:
            try:
                self.ep_sensor.unsub_distance()
            except Exception:
                pass
            self._dist_subscribed = False

    # -------- Gimbal scanning & wall sensing --------
    def read_distance_at(self, yaw_angle_deg, samples=5, timeout_s=1.0):
        """
        หมุนกิมบอลไปมุม yaw ที่ต้องการ แล้วอ่านระยะ (cm) แบบ median จากหลายตัวอย่าง
        คืนค่า: median distance (cm) หรือ None
        """
        self.last_distance_cm = None
        self.ep_gimbal.moveto(pitch=0, yaw=yaw_angle_deg, yaw_speed=180).wait_for_completed()

        distances = []
        self._sub_distance(freq=20)
        t0 = time.time()
        while len(distances) < samples and (time.time() - t0) < timeout_s:
            if self.last_distance_cm is not None:
                distances.append(self.last_distance_cm)
                self.last_distance_cm = None
            time.sleep(0.05)
        self._unsub_distance()

        if distances:
            med = float(np.median(distances))
            print(f"Median distance at yaw {yaw_angle_deg}: {med:.1f} cm (from {distances})")
            return med
        else:
            print(f"No valid distance readings at yaw {yaw_angle_deg}")
            return None

    def eye(self):
        """สแกนสามทิศ: F(0°), L(-90°), R(+90°) แล้วหันกลับหน้า"""
        print("Scanning: [F, L, R]")
        dist = {
            "F": self.read_distance_at(0),
            "L": self.read_distance_at(-90),
            "R": self.read_distance_at(90),
        }
        self.ep_gimbal.moveto(pitch=0, yaw=0, yaw_speed=180).wait_for_completed()
        return dist

    # -------- Body motion --------
    def turn(self, angle_deg):
        """หมุนตัวหุ่น angle_deg (deg). หมายเหตุ: z บวก = ทวนเข็ม → ใช้ -angle"""
        print(f"Action: Turning {angle_deg} degrees")
        self.ep_chassis.move(x=0, y=0, z=-angle_deg, z_speed=45).wait_for_completed()
        time.sleep(0.5)

    def move_forward_pid(self, cell_size_m,
                         Kp=1.2, Ki=0.05, Kd=0.1,
                         v_clip=0.7, tol_m=0.02):
        """
        เดินหน้า cell_size_m เมตร ด้วย PID ระยะจาก odometry:
        - ความเร็วหน้า = PID(distance_traveled)
        - ไม่คุม yaw ในฟังก์ชันนี้ (คง z=0) → เดินตรงดีในสภาพที่ล้อสมดุล
        """
        print(f"Action: Moving forward {cell_size_m} m using PID control.")
        pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=cell_size_m)

        sx, sy = self.get_xy_m()
        while True:
            x, y = self.get_xy_m()
            distance_traveled = math.hypot(x - sx, y - sy)
            speed = pid.compute(distance_traveled)
            speed = float(np.clip(speed, -v_clip, v_clip))
            self.ep_chassis.drive_speed(x=speed, y=0, z=0, timeout=0.1)

            error = cell_size_m - distance_traveled
            if abs(error) < tol_m:
                print("Target reached within tolerance.")
                break
            time.sleep(0.02)

        # stop and settle
        self.ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
        print("Movement complete.")
        time.sleep(0.5)

    # -------- Cleanup --------
    def close(self):
        try:
            self._unsub_distance()
            try:
                self.ep_chassis.unsub_attitude()
            except Exception:
                pass
            try:
                self.ep_chassis.unsub_position()
            except Exception:
                pass
            self.ep_robot.close()
        except Exception:
            pass
