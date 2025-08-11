# -*- coding:utf-8 -*-  # กำหนด encoding ของไฟล์เป็น UTF-8 เพื่อรองรับภาษาไทย
"""
RoboMaster Maze Control (Pond-only, Real Robot)
- ตัดสินใจตาม psuedo ของปอนด์ (if/else)
- สแกนกิมบอล L/F/R -> ตีความผนัง -> ALIGN -> เดิน 1 บล็อก
- ไม่มีส่วน Planner/Mapping และไม่มีโหมดจำลอง

สิ่งที่ต้องมีบนตัวหุ่น:
- TOF/Distance sensor อย่างน้อยด้านหน้า 1 ตัว ติดที่กิมบอล (หมุนซ้าย/ขวาได้)
- กิมบอลหมุน yaw ได้อย่างน้อย ±90°
"""

import time
import math

# ================== CONFIG (ปรับตามสนามจริง) ==================
GRID_R, GRID_C = 7, 7          # ขนาดกริดของเขาวงกต (ใช้แค่ log ตำแหน่งคร่าว ๆ)
CELL_LEN_M     = 0.30          # ความยาว 1 ช่อง (เมตร) *** วัดจริง
WALL_THR_M     = 0.22          # น้อยกว่านี้ = มีผนัง (ซ้าย/หน้า/ขวา) ***
STOP_THR_M     = 0.18          # หน้าใกล้กว่านี้ให้ stop ทันที ***
SCAN_DUR_S     = 0.10          # เวลารวมต่อมุมสำหรับเก็บ sample
SCAN_SAMPLES   = 8             # เก็บกี่ sample ต่อมุม (เพื่อหา median)
SETTLE_S       = 0.05          # หน่วงให้กิมบอลหยุดนิ่งก่อนอ่าน

# มุมกิมบอลสำหรับการสแกน (องศา)
YAW_LEFT  = -90.0
YAW_FRONT =   0.0
YAW_RIGHT = +90.0

# ความเร็ว/ข้อจำกัดการเคลื่อนที่
V_LINEAR_MPS   = 0.20          # m/s เดินหน้า
W_MAX_RADPS    = 1.20          # rad/s จำกัดความเร็วหมุน (จาก PID)
TURN_SPEED_DPS = 90.0          # deg/s ตอนสั่งหุ่นเลี้ยวด้วย chassis.move
MAX_CELL_TIME  = 5.0           # วินาที เดิน 1 ช่องนานสุด (กันค้าง)

# พารามิเตอร์ PD (ที่จริงเป็น PID แต่ปิด I ใช้ง่าย/ทน noise)
Kp = 1.2
Ki = 0.0
Kd = 0.06
DEADBAND = 0.01               # error เล็ก ๆ ให้เป็นศูนย์กัน jitter

# ตำแหน่ง/ทิศเริ่มต้น (สำหรับ log/หมุนสัมพัทธ์)
START_RC = (0, 0)             # (row, col) เริ่ม
START_HEADING_DEG = 0         # 0=N, 90=E, 180=S, 270=W

# ================== SDK IMPORT ==================
from robomaster import robot   # ใช้ SDK ของ RoboMaster

# ================== PID ==================
class PID:
    def __init__(self, Kp, Ki, Kd, out_limit=None, i_limit=None):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.out_limit = out_limit
        self.i_limit = i_limit
        self.reset()

    def reset(self):
        self.last_e = 0.0
        self.i = 0.0
        self.first = True

    def update(self, e, dt):
        if abs(e) < DEADBAND:
            e = 0.0
        if dt <= 0:
            dt = 1e-3
        self.i += e * dt
        if self.i_limit is not None:
            self.i = max(-self.i_limit, min(self.i, self.i_limit))
        d = 0.0 if self.first else (e - self.last_e) / dt
        self.first = False
        self.last_e = e
        u = self.Kp * e + self.Ki * self.i + self.Kd * d
        if self.out_limit is not None:
            u = max(-self.out_limit, min(u, self.out_limit))
        return u

# ================== Utils ==================
def wrap180(a):
    # แปลงองศาให้อยู่ในช่วง [-180, 180)
    return ((a + 180) % 360) - 180

def median(vals):
    s = sorted([v for v in vals if v is not None])
    n = len(s)
    if n == 0:
        return None
    return s[n//2] if n % 2 else 0.5 * (s[n//2 - 1] + s[n//2])

def heading_to_delta(heading_deg):
    # แปลงแนวทางเดิน -> การเปลี่ยน row/col
    h = int(round(heading_deg / 90.0)) % 4
    return [(-1, 0), (0, 1), (1, 0), (0, -1)][h]  # N,E,S,W

# ================== Robot Interface ==================
class RobotIf:
    """
    ห่อ SDK ให้เรียกใช้ง่าย/ปลอดภัยขึ้น:
    - เชื่อมต่อ, subscribe distance sensor (ถ้ามี)
    - เก็บ yaw ภายใน (ถ้ายังไม่ผูก IMU)
    """
    def __init__(self):
        self.ep = robot.Robot()
        self.ep.initialize(conn_type="ap")
        self.ep_chassis = self.ep.chassis
        self.ep_gimbal  = self.ep.gimbal
        self.ep_sensor  = getattr(self.ep, "sensor", None)

        # ใช้ heading ภายในไปก่อน (ถ้าอยากใช้ IMU ให้ subscribe ชุด attitude เพิ่ม)
        self.heading_deg_internal = START_HEADING_DEG

        # ----- Distance subscription -----
        self._tof_front_m = None     # ระยะด้านหน้าล่าสุด (เมตร) ที่กิมบอลกำลังชี้อยู่
        self._sub_started = False

        def _dist_cb(sub_info):
            """
            รูปแบบข้อมูลต่างกันตามรุ่น/SDK:
            - บางเครื่องส่ง int ระยะเดี่ยว (มม.)
            - บางเครื่องส่ง list/tuple หลายตัว [front, left, right, back] หรืออื่น ๆ (มม.)
            แก้ mapping ตรงนี้ให้ตรงกับเครื่องคุณ
            """
            try:
                if isinstance(sub_info, (list, tuple)):
                    # สมมติ index 0 = ตัวหน้าที่เราสนใจ (มม.)
                    mm = sub_info[0]
                else:
                    mm = int(sub_info)  # เดี่ยว (มม.)
                if mm is not None and mm > 0:
                    self._tof_front_m = mm / 1000.0
            except Exception:
                pass  # กันพังจากค่าประหลาด ๆ

        try:
            if self.ep_sensor is not None:
                self.ep_sensor.sub_distance(freq=10, callback=_dist_cb)
                self._sub_started = True
        except Exception:
            self._sub_started = False  # ไม่มี distance sensor/subscribe ไม่ผ่าน

        # (ทางเลือก) ถ้าอยากใช้ yaw จริง: เปิดคอมเมนต์เพื่อเก็บ yaw จาก IMU
        # self._yaw_deg = START_HEADING_DEG
        # try:
        #     def _att_cb(att):
        #         # att = {'pitch':..., 'roll':..., 'yaw':...} หรือคล้ายกันแล้วแต่ SDK
        #         y = att.get("yaw", None)
        #         if y is not None:
        #             self._yaw_deg = float(y)
        #     self.ep_chassis.sub_attitude(freq=10, callback=_att_cb)
        # except Exception:
        #     pass

    # ---------- Movement ----------
    def drive_vw(self, v_mps, wz_radps):
        # z ต้องเป็น deg/s
        self.ep_chassis.drive_speed(x=v_mps, y=0, z=math.degrees(wz_radps))

    def stop(self):
        self.ep_chassis.drive_speed(x=0, y=0, z=0)

    def turn_relative(self, delta_deg, z_speed_dps=TURN_SPEED_DPS):
        # เลี้ยวตัวหุ่นแบบสัมพัทธ์
        if abs(delta_deg) < 1.0:
            return
        self.ep_chassis.move(x=0, y=0, z=delta_deg, z_speed=z_speed_dps).wait_for_completed()
        # ถ้าไม่ได้ใช้ IMU ให้อัปเดต yaw ภายในเอง
        self.heading_deg_internal = (self.heading_deg_internal + delta_deg) % 360

    def turn_to_heading(self, target_deg, z_speed_dps=TURN_SPEED_DPS):
        cur = self.heading_deg_internal
        delta = wrap180(target_deg - cur)
        self.turn_relative(delta, z_speed_dps)

    # ---------- Gimbal ----------
    def gimbal_yaw_to(self, yaw_deg):
        # SDK บางรุ่นบล็อกเอง บางรุ่นต้อง wait
        self.ep_gimbal.moveto(yaw=yaw_deg, pitch=0).wait_for_completed()

    # ---------- Distance ----------
    def read_tof_m(self):
        """คืนค่าระยะด้านหน้าล่าสุด (เมตร) ที่กิมบอลกำลังชี้อยู่; ถ้าไม่มี => None"""
        return self._tof_front_m

    # ---------- Cleanup ----------
    def close(self):
        try:
            if getattr(self, "_sub_started", False) and self.ep_sensor is not None:
                try:
                    self.ep_sensor.unsub_distance()
                except Exception:
                    pass
            self.ep.close()
        except Exception:
            pass

# ================== Controller (Pond only) ==================
class NavigatorOnly:
    def __init__(self, rb: RobotIf):
        self.rb = rb
        self.pid_lane = PID(Kp, Ki, Kd, out_limit=W_MAX_RADPS, i_limit=0.5)
        self.rc = [START_RC[0], START_RC[1]]
        self.heading = START_HEADING_DEG
        self.last_scan = {'L': None, 'F': None, 'R': None}  # เก็บ median ล่าสุดของแต่ละทิศ

    # ---------- Scan ----------
    def scan_gimbal(self):
        """
        หมุนกิมบอลไป L-F-R แล้วอ่าน TOF หลายครั้งเพื่อหา median
        ผลลัพธ์:
        - walls: dict {'L':bool,'F':bool,'R':bool} (True=มีผนัง)
        - d:     dict {'L':m,'F':m,'R':m} ระยะ (เมตร) หรือ None
        """
        d = {}
        for label, yaw in (('L', YAW_LEFT), ('F', YAW_FRONT), ('R', YAW_RIGHT)):
            self.rb.gimbal_yaw_to(yaw)
            time.sleep(SETTLE_S)

            reads = []
            t0 = time.time()
            # เก็บ sample ภายในช่วงเวลา SCAN_DUR_S
            while time.time() - t0 < SCAN_DUR_S:
                dist = self.rb.read_tof_m()  # เมตร หรือ None
                if dist is not None:
                    reads.append(dist)
                time.sleep(SCAN_DUR_S / max(2, SCAN_SAMPLES))

            d[label] = median(reads) if reads else None

        walls = {k: (d[k] is not None and d[k] < WALL_THR_M) for k in ('L', 'F', 'R')}
        self.last_scan = d
        return walls, d

    # ---------- Decision ----------
    def decide_action(self, walls):
        """
        if/else ตาม psuedo เดิมของปอนด์
        ลำดับความสำคัญ: ถ้าหน้าโล่งและข้างเป็นผนังทั้งคู่ -> เดินหน้า
        จากนั้นเช็คกรณีต่าง ๆ
        """
        L, F, R = walls['L'], walls['F'], walls['R']
        if (not F) and L and R: return "FORWARD"
        if F and (not L) and R: return "TURN_LEFT"
        if F and L and (not R): return "TURN_RIGHT"
        if F and L and R:       return "UTURN"
        if not F: return "FORWARD"
        if not L: return "TURN_LEFT"
        if not R: return "TURN_RIGHT"
        return "UTURN"

    # ---------- Align (rotate chassis) ----------
    def align(self, action):
        if action == "FORWARD":
            return
        if action == "TURN_LEFT":
            self.heading = (self.heading - 90) % 360
        elif action == "TURN_RIGHT":
            self.heading = (self.heading + 90) % 360
        elif action == "UTURN":
            self.heading = (self.heading + 180) % 360
        self.rb.turn_to_heading(self.heading, z_speed_dps=TURN_SPEED_DPS)

    # ---------- Move one cell ----------
    def move_one_cell(self):
        """
        เดินหน้า 1 ช่อง
        - ใช้ค่า L/R จากสแกนล่าสุดเพื่อประมาณ "กลางเลน" (ไม่มีการสแกนขณะวิ่ง)
        - ถ้าหน้าเข้าใกล้ STOP_THR_M ให้หยุด-สแกนใหม่ (กันชน)
        """
        self.pid_lane.reset()
        start = time.time()
        moved_m = 0.0
        dt = 0.02

        # อัปเดตกิมบอลให้หันหน้าระหว่างวิ่ง เพื่อเก็บ front TOF เผื่อฉุกเฉิน
        self.rb.gimbal_yaw_to(YAW_FRONT)
        time.sleep(SETTLE_S)

        # ค่าฐาน L/R จากการสแกนล่าสุด (อาจเป็น None ถ้าอ่านไม่ได้)
        base_L = self.last_scan.get('L', None)
        base_R = self.last_scan.get('R', None)

        while moved_m < CELL_LEN_M:
            # error: อยากให้ dL == dR (อยู่กึ่งกลาง)
            e = 0.0
            dL = base_L
            dR = base_R
            if (dL is not None) and (dR is not None):
                e = (dR - dL)  # ขวาไกลกว่า => หมุนขวาเล็กน้อยให้กลับกลาง

            w = self.pid_lane.update(e, dt)
            w = max(-W_MAX_RADPS, min(w, W_MAX_RADPS))

            # กันชนด้านหน้า
            dF_now = self.rb.read_tof_m()
            if (dF_now is not None) and (dF_now < STOP_THR_M):
                self.rb.stop()
                walls, _ = self.scan_gimbal()   # รีสแกนเพื่ออัปเดตสถานะจริง
                return "blocked"

            # ขับไปข้างหน้า
            self.rb.drive_vw(V_LINEAR_MPS, w)

            moved_m = (time.time() - start) * V_LINEAR_MPS
            if time.time() - start > MAX_CELL_TIME:
                self.rb.stop()
                return "timeout"

            time.sleep(dt)

        # ครบ 1 ช่อง
        self.rb.stop()
        dr, dc = heading_to_delta(self.heading)
        self.rc[0] = max(0, min(GRID_R - 1, self.rc[0] + dr))
        self.rc[1] = max(0, min(GRID_C - 1, self.rc[1] + dc))
        return "ok"

    # ---------- Main loop ----------
    def run(self, max_steps=200):
        steps = 0
        try:
            while steps < max_steps:
                walls, d = self.scan_gimbal()
                print(f"[SCAN] cell={tuple(self.rc)} head={self.heading}  d={d}  walls={walls}")

                action = self.decide_action(walls)
                print(f"[DECIDE] -> {action}")

                self.align(action)

                if action == "UTURN":
                    # ยูเทิร์นเสร็จ ให้สแกนใหม่รอบหน้า
                    steps += 1
                    continue

                res = self.move_one_cell()
                print(f"[MOVE] result={res} new_cell={tuple(self.rc)}")
                steps += 1
        finally:
            self.rb.stop()

# ================== main ==================
def main():
    rb = RobotIf()
    nav = NavigatorOnly(rb)
    try:
        nav.run(max_steps=200)
    finally:
        rb.close()

if __name__ == "__main__":
    main()
