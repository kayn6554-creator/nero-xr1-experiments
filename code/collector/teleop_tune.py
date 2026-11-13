#!/usr/bin/env python3
# 遥操作调参版：teleop_tune.py [时长s] [关节单步rad] [夹爪单步m]
# 在 teleop_gripper.py 基础上增加延迟来源统计；安全逻辑不变
import math, sys, time, rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Empty
from pyAgxArm import create_agx_arm_config, AgxArmFactory, NeroFW

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
STEP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
GRIP_STEP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002
if not (0 < STEP <= 0.06): raise SystemExit("关节单步只允许 (0, 0.06] rad")
if not (0 < GRIP_STEP <= 0.005): raise SystemExit("夹爪单步只允许 (0, 0.005] m")
RATE = 50.0; ALIGN_DEG = 3.0; DIVERGE_DEG = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0; STALE_S = 0.5
if not (0 < DIVERGE_DEG <= 20): raise SystemExit("偏差上限只允许 (0, 20]°")
LEADER_GRIP_MAX = 0.0105; FOLLOW_GRIP_MAX = 0.100; GRIP_FORCE = 1.0
MARGIN = math.radians(1.0)
LIM = [(-155,155),(-100,100),(-158,158),(-58,123),(-158,158),(-42,55),(-80,80)]
LIM = [(math.radians(a) + MARGIN, math.radians(b) - MARGIN) for a, b in LIM]
names = [f"joint{i}" for i in range(1, 8)]

leader = AgxArmFactory.create_arm(create_agx_arm_config(robot="nero", comm="can", channel="can1", firmeware_version=NeroFW.V112))
leader.connect()
lgrip = leader.init_effector(leader.OPTIONS.EFFECTOR.AGX_GRIPPER)
rclpy.init(signal_handler_options=SignalHandlerOptions.NO); n = Node("panhe_teleop_tune")
fol = {}; fol_t = [0.0]; fol_n = [0]
def on_js(m):
    fol.update(dict(zip(m.name, m.position))); fol_t[0] = time.monotonic(); fol_n[0] += 1
n.create_subscription(JointState, "/feedback/joint_states", on_js, 10)
pub_arm = n.create_publisher(JointState, "/control/move_js", 1)
pub_grip = n.create_publisher(JointState, "/control/joint_states", 1)
gate = n.create_client(SetBool, "/control_enable"); hold = n.create_client(Empty, "/emergency_stop")
gate.wait_for_service(timeout_sec=3); hold.wait_for_service(timeout_sec=3)
def call(c, r):
    f = c.call_async(r); rclpy.spin_until_future_complete(n, f, timeout_sec=2); return f.result()
def set_gate(v):
    r = SetBool.Request(); r.data = v; res = call(gate, r); print("控制锁:", res.message if res else "无响应")
def leader_grip_target():
    g = lgrip.get_gripper_ctrl_states()
    if g is None: return None
    return min(max(g.msg.value / LEADER_GRIP_MAX * FOLLOW_GRIP_MAX, 0.0), FOLLOW_GRIP_MAX)
def pct(xs, p):
    s = sorted(xs); return s[min(len(s) - 1, int(p * len(s)))]

t0 = time.time(); la = None
while time.time() - t0 < 3 and (la is None or len(fol) < 8):
    rclpy.spin_once(n, timeout_sec=0.1); la = leader.get_leader_joint_angles()
if la is None or len(fol) < 8 or "gripper" not in fol:
    print("读不到主臂/从臂/夹爪数据，取消"); raise SystemExit
diffs = [math.degrees(la.msg[i] - fol[k]) for i, k in enumerate(names)]
print("启动前关节差值(°):", [round(d, 1) for d in diffs])
if max(abs(d) for d in diffs) > ALIGN_DEG:
    print(f"有关节差值超过 {ALIGN_DEG}°，请先运行 align_follower.py。取消。"); raise SystemExit
print(f"\n运行 {DURATION:.0f} 秒 | 关节最快 {math.degrees(STEP*RATE):.0f}°/s | 夹爪最快 {GRIP_STEP*RATE*1000:.0f} mm/s")
if input("手在急停上、夹爪附近无手指和物体？输入 yes 开始：").strip() != "yes":
    print("已取消"); raise SystemExit

cmd = [fol[k] for k in names]; gcmd = fol["gripper"]
lags, devs, tots = [], [], []; cycles = 0
reason = "时间到"
set_gate(True)
t_start = time.monotonic(); fol_n[0] = 0
try:
    last_print = 0.0; last_lt = None; last_lt_change = time.monotonic()
    while time.monotonic() - t_start < DURATION:
        tick = time.monotonic()
        rclpy.spin_once(n, timeout_sec=0.0)
        la = leader.get_leader_joint_angles(); now = time.monotonic()
        if la is None: reason = "主臂无数据"; break
        if la.timestamp != last_lt: last_lt = la.timestamp; last_lt_change = now
        if now - last_lt_change > STALE_S: reason = "主臂数据超时"; break
        if now - fol_t[0] > STALE_S: reason = "从臂数据超时"; break
        goals = [min(max(la.msg[i], LIM[i][0]), LIM[i][1]) for i in range(7)]
        for i in range(7):
            cmd[i] += max(-STEP, min(STEP, goals[i] - cmd[i]))
        lag = max(abs(math.degrees(goals[i] - cmd[i])) for i in range(7))
        dev = max(abs(math.degrees(cmd[i] - fol[k])) for i, k in enumerate(names))
        tot = max(abs(math.degrees(goals[i] - fol[k])) for i, k in enumerate(names))
        lags.append(lag); devs.append(dev); tots.append(tot); cycles += 1
        if dev > DIVERGE_DEG: reason = f"从臂跟踪误差 {dev:.1f}° 过大"; break
        m = JointState(); m.name = names; m.position = list(cmd); pub_arm.publish(m)
        gt = leader_grip_target()
        if gt is not None:
            gcmd += max(-GRIP_STEP, min(GRIP_STEP, gt - gcmd))
            g = JointState(); g.name = ["gripper"]; g.position = [gcmd]; g.effort = [GRIP_FORCE]; pub_grip.publish(g)
        if now - last_print > 0.5:
            print(f"\r剩余 {DURATION-(now-t_start):4.0f}s | A 限幅滞后 {lag:5.1f}° | B 跟踪误差 {dev:5.1f}° | 夹爪 {gcmd*1000:5.1f}/{fol['gripper']*1000:5.1f} mm  ", end="", flush=True)
            last_print = now
        time.sleep(max(0.0, 1.0 / RATE - (time.monotonic() - tick)))
except KeyboardInterrupt:
    reason = "Ctrl+C"
finally:
    el = time.monotonic() - t_start
    print(f"\n停止原因：{reason}")
    set_gate(False)
    call(hold, Empty.Request()); print("已发送原地保持")
    if cycles:
        print(f"STEP={STEP} rad | 实际循环 {cycles/el:.1f} Hz | 从臂反馈 {fol_n[0]/el:.1f} Hz")
        for name, xs in [("A 限幅滞后 (主臂目标-指令)", lags), ("B 跟踪误差 (指令-从臂实际)", devs), ("总偏差   (主臂目标-从臂)", tots)]:
            print(f"{name}: 平均 {sum(xs)/len(xs):5.2f}° | P95 {pct(xs, .95):5.2f}° | 最大 {max(xs):5.2f}°")
    leader.disconnect(); n.destroy_node(); rclpy.shutdown()
