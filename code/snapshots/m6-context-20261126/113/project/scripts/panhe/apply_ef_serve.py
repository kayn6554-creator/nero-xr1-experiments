"""Let scripts/serve_openpi_policy.py serve EF (action_layout nero_ee) models (run from the xr1 dir on 113).

The EF model predicts, per step, the flange pose relative to the chunk-start flange
(pos 0-2, axis-angle 3-5), the gripper (6) and the j3 delta (7).  The server turns
that into the SAME reply as a nero_joint model -- 30 x [7 absolute joints, gripper]
-- by running our IK (mibot/utils/nero_kin.py; seed = current state, then the
previous step; j3 target = state j3 + predicted delta).  So the client / testbench
needs no change.  Safety: any step whose IK does not converge, or whose joints jump
more than NERO_EE_MAX_JUMP_DEG (default 15) from the previous step, makes the whole
request fail (the client then refuses to execute).  The reply also carries
"ik": {max_pos_err_mm, max_rot_err_deg, max_step_jump_deg, ms}.

nero_joint serving is untouched.  Every snippet must match exactly once, else
nothing is written.  Backup: scripts/serve_openpi_policy.py.before_ef
Needs mibot/utils/nero_kin.py (apply_ef.py copies it there).
"""
import shutil
from pathlib import Path

PATH = "scripts/serve_openpi_policy.py"

FUNC = '''

def nero_ee_to_joint(delta: np.ndarray, state8: np.ndarray, gripper_absolute: bool):
    """EF: [T,60] un-normalized nero_ee actions -> [T,8] absolute joints + gripper via IK."""
    import os
    import time

    from mibot.utils import nero_kin as K

    delta, state8 = np.asarray(delta, dtype=np.float64), np.asarray(state8, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 60 or state8.shape != (8,):
        raise ValueError("Nero EF requires delta [T,60] and state [8]")
    if not np.isfinite(delta).all() or not np.isfinite(state8).all():
        raise ValueError("state/actions contain NaN or Inf")
    max_jump = float(os.environ.get("NERO_EE_MAX_JUMP_DEG", "15"))
    t0 = time.perf_counter()
    T0 = K.fk(state8[:7])
    q_prev = state8[:7].copy()
    out = np.zeros((len(delta), 8), dtype=np.float32)
    worst = {"max_pos_err_mm": 0.0, "max_rot_err_deg": 0.0, "max_step_jump_deg": 0.0}
    for k in range(len(delta)):
        target = np.eye(4)
        target[:3, :3] = T0[:3, :3] @ K.rot_exp(delta[k, 3:6])
        target[:3, 3] = T0[:3, 3] + T0[:3, :3] @ delta[k, 0:3]
        q, ok, ep, er = K.ik(target, q_prev, q3_target=state8[2] + delta[k, 7])
        jump = float(np.degrees(np.abs(q - q_prev).max()))
        if not ok:
            raise ValueError(f"EF IK failed at step {k}: pos err {ep * 1000:.2f} mm, rot err {np.degrees(er):.2f} deg")
        if jump > max_jump:
            raise ValueError(f"EF IK joint jump {jump:.1f} deg at step {k} > {max_jump} deg")
        worst["max_pos_err_mm"] = max(worst["max_pos_err_mm"], float(ep) * 1000)
        worst["max_rot_err_deg"] = max(worst["max_rot_err_deg"], float(np.degrees(er)))
        worst["max_step_jump_deg"] = max(worst["max_step_jump_deg"], jump)
        out[k, :7] = q
        q_prev = q
    out[:, 7] = delta[:, 6] if gripper_absolute else delta[:, 6] + state8[7]
    worst = {key: round(float(value), 4) for key, value in worst.items()}
    worst["ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return out, worst
'''

EDITS = [
    ('    return delta[:, :8] + state8[None, :]\n',
     '    return delta[:, :8] + state8[None, :]\n' + FUNC),
    ('        if action_layout not in {"piper_joint", "piper_left_joint", "nero_joint"}:\n',
     '        if action_layout not in {"piper_joint", "piper_left_joint", "nero_joint", "nero_ee"}:\n'),
    ('        self.action_dim = 8 if action_layout == "nero_joint" else 14\n'
     '        self.CAMERA_KEYS = ("front", "wrist") if action_layout == "nero_joint" else type(self).CAMERA_KEYS\n',
     '        self.action_dim = 8 if action_layout in ("nero_joint", "nero_ee") else 14\n'
     '        self.CAMERA_KEYS = ("front", "wrist") if action_layout in ("nero_joint", "nero_ee") else type(self).CAMERA_KEYS\n'),
    ('            "state_layout": "joint7,gripper" if action_layout == "nero_joint" else',
     '            "state_layout": "joint7,gripper" if action_layout in ("nero_joint", "nero_ee") else'),
    ('        if action_layout == "nero_joint":\n            self.metadata["rtc"] = {"mode": "disabled", "max_prefix_steps": 0}\n',
     '        if action_layout in ("nero_joint", "nero_ee"):\n            self.metadata["rtc"] = {"mode": "disabled", "max_prefix_steps": 0}\n'),
    ('        if action_layout == "nero_joint":\n            front, wrist = images\n',
     '        if action_layout in ("nero_joint", "nero_ee"):\n            front, wrist = images\n'),
    ('        dimension = {"nero_joint": 8, "piper_left_joint": 7, "piper_joint": 14}[layout]\n',
     '        dimension = {"nero_joint": 8, "nero_ee": 8, "piper_left_joint": 7, "piper_joint": 14}[layout]\n'),
    ('        if layout == "nero_joint" and "rtc_prefix" in obs:\n',
     '        if layout in ("nero_joint", "nero_ee") and "rtc_prefix" in obs:\n'),
    ('        if layout == "nero_joint":\n            camera_keys = ("front", "wrist")\n',
     '        if layout in ("nero_joint", "nero_ee"):\n            camera_keys = ("front", "wrist")\n'),
    ('        if layout == "nero_joint":\n            packed_state = np.zeros((1, 60), dtype=np.float32)\n',
     '        if layout in ("nero_joint", "nero_ee"):\n            packed_state = np.zeros((1, 60), dtype=np.float32)\n'),
    ('        delta = delta[0].float().cpu().numpy()\n',
     '        delta = delta[0].float().cpu().numpy()\n'
     '        if getattr(self, "action_layout", "") == "nero_ee":\n'
     '            actions, ik_report = nero_ee_to_joint(delta, state14, getattr(self, "gripper_absolute", False))\n'
     '            return {"actions": actions, "prompt_used": getattr(self, "last_prompt", None), "ik": ik_report}\n'),
]

if not Path("mibot/utils/nero_kin.py").exists():
    raise SystemExit("mibot/utils/nero_kin.py missing (run apply_ef.py first), nothing written")
text = Path(PATH).read_text()
if "def nero_ee_to_joint" in text:
    raise SystemExit("already applied, nothing written")
for old, new in EDITS:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{PATH}: expected 1 match, found {n}:\n{old}")
    text = text.replace(old, new)
if not Path(PATH + ".before_ef").exists():
    shutil.copy2(PATH, PATH + ".before_ef")
Path(PATH).write_text(text)
print("edited", PATH, "(backup .before_ef)")