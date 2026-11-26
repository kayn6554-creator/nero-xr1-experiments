"""M5 (step flag, 11-12 version A): add action layout "nero_joint_sb" (run from /mnt/panhe/xr1).

nero_joint_sb = nero_joint (slots 0-6 joint deltas, 7 gripper) + slot 8 = step flag:
for each of the 30 future frames, 1 if that frame already belongs to the NEXT step
(source.step_index differs from the chunk-start frame), else 0.  Normalised with the
data config's mean/std for slot 8 (make_m5_config.py sets 0.5 / 0.5 -> -1 / +1).
Action targets for slots 0-7 are exactly as nero_joint (natural, cross the boundary);
only the extra slot is learned.  At inference the server can cut the chunk where the
predicted flag turns on (hold the arm there) and later advance the step text itself.
Other layouts unchanged.  Backups: <file>.before_m5sb
"""
import shutil
from pathlib import Path

FLAG = '''            if self.action_layout == "nero_joint_sb":
                # M5: slot 8 = 1 where a future frame already belongs to the next step.
                idx = (traj.get("source") or {}).get("step_index")
                flag = np.zeros((steps, 1), dtype=np.float32)
                if idx is not None:
                    flag[:, 0] = [float(idx[frame + k] != idx[frame]) for k in range(steps)]
                flag = self._pad(flag, steps)
                action[:, 8:9] = (flag - self.mean[:, 8:9]) / (self.std[:, 8:9] + 1e-6)
'''
EDITS = {
    "mibot/utils/io.py": [
        ('    "nero_joint": NERO_JOINT_ACTION_PARTS,\n',
         '    "nero_joint": NERO_JOINT_ACTION_PARTS,\n'
         '    # M5: nero_joint + slot 8 step flag (1 = frame already in the next step).\n'
         '    "nero_joint_sb": NERO_JOINT_ACTION_PARTS + (("step_flag", slice(8, 9)),),\n'),
    ],
    "mibot/data/datasets/json_dataset.py": [
        ('        elif self.action_layout == "nero_joint":\n',
         '        elif self.action_layout in ("nero_joint", "nero_joint_sb"):\n'),
        ('            action[:, :7], action[:, 7:8] = joint, gripper\n'
         '        elif self.action_layout == "nero_ee":\n',
         '            action[:, :7], action[:, 7:8] = joint, gripper\n'
         + FLAG +
         '        elif self.action_layout == "nero_ee":\n'),
        ('        if self.action_layout in ("nero_joint", "nero_ee"):\n',
         '        if self.action_layout in ("nero_joint", "nero_ee", "nero_joint_sb"):\n'),
    ],
}
new = {}
for path, edits in EDITS.items():
    text = Path(path).read_text()
    if "nero_joint_sb" in text:
        raise SystemExit(f"{path}: already applied, nothing written")
    for old, rep in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{path}: expected 1 match, found {n}:\n{old}")
        text = text.replace(old, rep)
    new[path] = text
for path, text in new.items():
    shutil.copy2(path, path + ".before_m5sb")
    Path(path).write_text(text)
    print("edited", path, "(backup .before_m5sb)")