"""Add unified action_layout "nero_ee_sb" (EF + M5 Step Flag + Unified EE State)."""
from __future__ import annotations

import shutil
from pathlib import Path

EE_SB_BRANCH = '''        elif self.action_layout == "nero_ee_sb":
            if is_piper_traj(traj):
                q0 = self._frame(traj, "proprios.left_arm_joint", frame).astype(np.float64)
                qa = self._future(traj, "actions.left_arm_joint", frame, steps).astype(np.float64)
                T0 = piper_flange_pose(q0)
                Ts = piper_flange_poses(qa)
                pos, aa = ee_relative_action(T0, Ts)
                grip = self._future(traj, "actions.left_gripper_pos", frame, steps).astype(np.float64).reshape(steps, 1) * PIPER_GRIPPER_TO_NERO_M
                if not self.gripper_absolute:
                    g0 = float(self._frame(traj, "proprios.left_gripper_pos", frame).reshape(-1)[0]) * PIPER_GRIPPER_TO_NERO_M
                    grip = grip - g0
                elbow = np.zeros((steps, 1), dtype=np.float32)
                has_elbow = False
            else:
                q0 = self._frame(traj, "proprios.arm_joint", frame).astype(np.float64)
                qa = self._future(traj, "actions.arm_joint", frame, steps).astype(np.float64)
                T0 = nero_flange_pose(q0)
                Ts = nero_flange_poses(qa)
                pos, aa = ee_relative_action(T0, Ts)
                grip = self._future(traj, "actions.gripper_pos", frame, steps).astype(np.float64).reshape(steps, 1)
                if not self.gripper_absolute:
                    g0 = float(self._frame(traj, "proprios.gripper_pos", frame).reshape(-1)[0])
                    grip = grip - g0
                elbow = qa[:, 2:3] - q0[2]
                has_elbow = True
            idx = (traj.get("source") or {}).get("step_index")
            has_step_flag = idx is not None
            flag = np.zeros((steps, 1), dtype=np.float32)
            if has_step_flag:
                flag[:, 0] = [float(idx[frame + k] != idx[frame]) for k in range(steps)]
            values = tuple(self._pad(v, steps) for v in (pos, aa, grip, elbow, flag))
            parts = normalize_action_parts(values, self.mean, self.std, self.action_layout)
            action = np.zeros((self.action_length, 60), dtype=np.float32)
            for (_, slc), value in zip(get_action_parts(self.action_layout), parts):
                if value.shape != (self.action_length, slc.stop - slc.start):
                    raise ValueError(f"nero_ee_sb part {slc} has shape {value.shape}")
                action[:, slc] = value
'''

EDITS = {
    "mibot/utils/io.py": [
        (
            'NERO_EE_ACTION_PARTS = (\n'
            '    ("ee_pos", slice(0, 3)), ("ee_aa", slice(3, 6)), ("gripper", slice(6, 7)), ("elbow", slice(7, 8)),\n'
            ')\n',
            'NERO_EE_ACTION_PARTS = (\n'
            '    ("ee_pos", slice(0, 3)), ("ee_aa", slice(3, 6)), ("gripper", slice(6, 7)), ("elbow", slice(7, 8)),\n'
            ')\n'
            'NERO_EE_SB_ACTION_PARTS = NERO_EE_ACTION_PARTS + (("step_flag", slice(8, 9)),)\n',
        ),
        (
            '    "nero_ee": NERO_EE_ACTION_PARTS,\n',
            '    "nero_ee": NERO_EE_ACTION_PARTS,\n'
            '    "nero_ee_sb": NERO_EE_SB_ACTION_PARTS,\n',
        ),
    ],
    "mibot/data/datasets/json_dataset.py": [
        (
            'from mibot.utils.nero_kin import fk as nero_fk, rot_log_batch as nero_rot_log_batch\n',
            'from mibot.utils.nero_kin import fk as nero_fk, rot_log_batch as nero_rot_log_batch\n'
            'from mibot.utils.ef_align import (\n'
            '    PIPER_GRIPPER_TO_NERO_M,\n'
            '    compose_ee_state,\n'
            '    ee_relative_action,\n'
            '    is_piper_traj,\n'
            '    nero_flange_pose,\n'
            '    nero_flange_poses,\n'
            '    piper_flange_pose,\n'
            '    piper_flange_poses,\n'
            ')\n',
        ),
        (
            '                if value.shape != (self.action_length, slc.stop - slc.start):\n'
            '                    raise ValueError(f"nero_ee part {slc} has shape {value.shape}")\n'
            '                action[:, slc] = value\n'
            '        else:\n'
            '            raise ValueError(f"unsupported action_layout={self.action_layout!r}")',
            '                if value.shape != (self.action_length, slc.stop - slc.start):\n'
            '                    raise ValueError(f"nero_ee part {slc} has shape {value.shape}")\n'
            '                action[:, slc] = value\n'
            + EE_SB_BRANCH.rstrip("\n")
            + '\n        else:\n'
            '            raise ValueError(f"unsupported action_layout={self.action_layout!r}")',
        ),
        (
            '        action_mask = build_action_mask(\n'
            '            self.action_length, self._mask(traj, steps), self.action_layout\n'
            '        )\n',
            '        action_mask = build_action_mask(\n'
            '            self.action_length, self._mask(traj, steps), self.action_layout\n'
            '        )\n'
            '        if self.action_layout == "nero_ee_sb":\n'
            '            if not has_elbow:\n'
            '                action_mask[:, 7:8] = 0\n'
            '            if not has_step_flag:\n'
            '                action_mask[:, 8:9] = 0\n',
        ),
        (
            '        traj = self._read_json(path)\n'
            '        q = np.asarray(get_value(traj, "proprios.arm_joint"), dtype=np.float64)\n'
            '        g = np.asarray(get_value(traj, "actions.gripper_pos"), dtype=np.float64).reshape(len(q), -1)[:, 0]\n',
            '        traj = self._read_json(path)\n'
            '        if is_piper_traj(traj):\n'
            '            q = np.asarray(get_value(traj, "proprios.left_arm_joint"), dtype=np.float64)\n'
            '            g = np.asarray(get_value(traj, "actions.left_gripper_pos"), dtype=np.float64).reshape(len(q), -1)[:, 0] * PIPER_GRIPPER_TO_NERO_M\n'
            '        else:\n'
            '            q = np.asarray(get_value(traj, "proprios.arm_joint"), dtype=np.float64)\n'
            '            g = np.asarray(get_value(traj, "actions.gripper_pos"), dtype=np.float64).reshape(len(q), -1)[:, 0]\n',
        ),
        (
            '        if self.action_layout == "piper_left_joint" and "observations.wrist_right" in prompt["images"]:\n',
            '        if self.action_layout in ("piper_left_joint", "nero_ee_sb") and "observations.wrist_right" in prompt["images"]:\n',
        ),
        (
            '            if getattr(self, "step_prompt", False):\n'
            '                conversations[0]["value"] += " " + step_text_for_frame(\n'
            '                    traj, frame, getattr(self, "step_merge23", False))\n',
            '            if getattr(self, "step_prompt", False):\n'
            '                src = traj.get("source") or {}\n'
            '                if src.get("step_index") is not None and str(src.get("dataset", "")).startswith("nero_"):\n'
            '                    conversations[0]["value"] += " " + step_text_for_frame(\n'
            '                        traj, frame, getattr(self, "step_merge23", False))\n',
        ),
        (
            '        if self.action_layout in ("nero_joint", "nero_ee", "nero_joint_sb"):\n',
            '        if self.action_layout == "nero_ee_sb":\n'
            '            if is_piper_traj(traj):\n'
            '                T0 = piper_flange_pose(self._frame(traj, "proprios.left_arm_joint", frame))\n'
            '                g0 = float(self._frame(traj, "proprios.left_gripper_pos", frame).reshape(-1)[0]) * PIPER_GRIPPER_TO_NERO_M\n'
            '                return compose_ee_state(T0, g0, is_piper=True)\n'
            '            T0 = nero_flange_pose(self._frame(traj, "proprios.arm_joint", frame))\n'
            '            g0 = float(self._frame(traj, "proprios.gripper_pos", frame).reshape(-1)[0])\n'
            '            return compose_ee_state(T0, g0, is_piper=False)\n'
            '        if self.action_layout in ("nero_joint", "nero_ee", "nero_joint_sb"):\n',
        ),
    ],
}

def main():
    for rel, pairs in EDITS.items():
        p = Path(rel)
        txt = p.read_text()
        if "nero_ee_sb" in txt:
            print(f"[skip] {rel} already contains nero_ee_sb")
            continue
        for old, _ in pairs:
            cnt = txt.count(old)
            assert cnt == 1, f"{rel}: expected 1 match, got {cnt} for:\n{old}"
        bak = p.with_suffix(p.suffix + ".before_ee_sb")
        if not bak.exists():
            shutil.copy2(p, bak)
            print(f"[backup] {bak}")
        for old, new in pairs:
            txt = txt.replace(old, new, 1)
        p.write_text(txt)
        print(f"[patched] {rel}")

    src_m = Path("configs/model/posttrain_nero_ee_full.yaml")
    dst_m = Path("configs/model/posttrain_nero_ee_sb_full.yaml")
    if not dst_m.exists():
        mtxt = src_m.read_text()
        assert mtxt.count("      action_layout: nero_ee\n") == 1
        dst_m.write_text(mtxt.replace("      action_layout: nero_ee\n", "      action_layout: nero_ee_sb\n", 1))
        print(f"[wrote] {dst_m}")
    else:
        print(f"[skip] {dst_m} already exists")

if __name__ == "__main__":
    main()