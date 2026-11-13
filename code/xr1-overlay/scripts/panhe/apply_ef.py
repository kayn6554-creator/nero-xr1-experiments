"""Add the EF (end-effector) action layout "nero_ee" (run from /mnt/panhe/xr1).

nero_ee action, all relative to the chunk start (follower state at frame t):
  slots 0-2  flange position delta in the chunk-start flange frame   R0^T (p_k - p0)
  slots 3-5  flange rotation delta, axis-angle                        aa(R0^T R_k)
  slot  6    gripper (absolute target if gripper_absolute, else delta) -- XR-1's own gripper slot
  slot  7    elbow: j3_k - j3_0 (fixes the 7-DoF redundancy so IK reproduces the demo)
Flange poses come from pyAgxArm's MDH FK of the recorded command joints
(mibot/utils/nero_kin.py, a numpy copy checked against pyAgxArm.fk_from_mdh).
State is unchanged (7 joints + gripper, same q01/q99 as nero_joint).

Also writes configs/model/posttrain_nero_ee_full.yaml (= nero joint model config
with action_layout nero_ee).  nero_joint behaviour is untouched.
Every snippet must match exactly once, otherwise nothing is written.
Backups: <file>.before_ef
"""
import shutil
from pathlib import Path

EE_BRANCH = '''        elif self.action_layout == "nero_ee":
            # EF: flange pose of the commanded joints relative to the follower's
            # flange at the chunk start, + gripper, + j3 delta (elbow).
            q0 = self._frame(traj, "proprios.arm_joint", frame).astype(np.float64)
            qa = self._future(traj, "actions.arm_joint", frame, steps).astype(np.float64)
            T0 = nero_fk(q0)
            Ts = np.stack([nero_fk(q) for q in qa])
            R0t = T0[:3, :3].T
            pos = (R0t @ (Ts[:, :3, 3] - T0[:3, 3]).T).T
            aa = nero_rot_log_batch(np.einsum("ij,njk->nik", R0t, Ts[:, :3, :3]))
            elbow = qa[:, 2:3] - q0[2]
            grip = self._future(traj, "actions.gripper_pos", frame, steps)
            if not self.gripper_absolute:
                grip = grip - self._frame(traj, "proprios.gripper_pos", frame)
            values = tuple(self._pad(v, steps) for v in (pos, aa, grip, elbow))
            parts = normalize_action_parts(values, self.mean, self.std, self.action_layout)
            action = np.zeros((self.action_length, 60), dtype=np.float32)
            for (_, slc), value in zip(get_action_parts(self.action_layout), parts):
                if value.shape != (self.action_length, slc.stop - slc.start):
                    raise ValueError(f"nero_ee part {slc} has shape {value.shape}")
                action[:, slc] = value
'''

EDITS = {
    "mibot/utils/io.py": [
        (
            'NERO_JOINT_ACTION_PARTS = (("joint", slice(0, 7)), ("gripper", slice(7, 8)))\n',
            'NERO_JOINT_ACTION_PARTS = (("joint", slice(0, 7)), ("gripper", slice(7, 8)))\n'
            '# EF: flange pos delta, axis-angle delta, gripper (XR-1 slot 6), j3 delta (elbow).\n'
            'NERO_EE_ACTION_PARTS = (\n'
            '    ("ee_pos", slice(0, 3)), ("ee_aa", slice(3, 6)), ("gripper", slice(6, 7)), ("elbow", slice(7, 8)),\n'
            ')\n',
        ),
        (
            '    "nero_joint": NERO_JOINT_ACTION_PARTS,\n',
            '    "nero_joint": NERO_JOINT_ACTION_PARTS,\n'
            '    "nero_ee": NERO_EE_ACTION_PARTS,\n',
        ),
    ],
    "mibot/data/datasets/json_dataset.py": [
        (
            '    compose_state,\n    get_value,\n',
            '    compose_state,\n    get_action_parts,\n    get_value,\n',
        ),
        (
            'from mibot.utils.nero_steps import step_text_for_frame\n',
            'from mibot.utils.nero_steps import step_text_for_frame\n'
            'from mibot.utils.nero_kin import fk as nero_fk, rot_log_batch as nero_rot_log_batch\n',
        ),
        (
            '            action[:, :7], action[:, 7:8] = joint, gripper\n'
            '        else:\n'
            '            raise ValueError(f"unsupported action_layout={self.action_layout!r}")\n',
            '            action[:, :7], action[:, 7:8] = joint, gripper\n'
            + EE_BRANCH +
            '        else:\n'
            '            raise ValueError(f"unsupported action_layout={self.action_layout!r}")\n',
        ),
        (
            '        if self.action_layout == "nero_joint":\n'
            '            joint = self._frame(traj, "proprios.arm_joint", frame)\n',
            '        if self.action_layout in ("nero_joint", "nero_ee"):\n'
            '            joint = self._frame(traj, "proprios.arm_joint", frame)\n',
        ),
    ],
}

MODEL_SRC = Path("configs/model/posttrain_nero_joint_full.yaml")
MODEL_DST = Path("configs/model/posttrain_nero_ee_full.yaml")
KIN_SRC = Path("scripts/panhe/nero_kin.py")
KIN_DST = Path("mibot/utils/nero_kin.py")

for p in (MODEL_DST, KIN_DST):
    if p.exists():
        raise SystemExit(f"{p} already exists (applied before?), nothing written")
model = MODEL_SRC.read_text()
if model.count("      action_layout: nero_joint\n") != 1:
    raise SystemExit(f"{MODEL_SRC}: action_layout line not found once, nothing written")
new_text = {}
for path, edits in EDITS.items():
    text = Path(path).read_text()
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{path}: expected 1 match, found {count}:\n{old}")
        text = text.replace(old, new)
    new_text[path] = text
for path, text in new_text.items():
    if not Path(path + ".before_ef").exists():
        shutil.copy2(path, path + ".before_ef")
    Path(path).write_text(text)
    print("edited", path, "(backup .before_ef)")
shutil.copy2(KIN_SRC, KIN_DST)
print("copied", KIN_SRC, "->", KIN_DST)
MODEL_DST.write_text(model.replace("      action_layout: nero_joint\n", "      action_layout: nero_ee\n"))
print("wrote", MODEL_DST)
