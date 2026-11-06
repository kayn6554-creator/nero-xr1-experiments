"""Apply the M2 edits on top of M1 (run from /mnt/panhe/xr1).

M2 = (a) gripper target is the recorded command (absolute) instead of
command - current opening; (b) long static runs are thinned as training
anchors: keep the first `head` and last `tail` frames of each run, drop the
middle. Both are off unless the data config turns them on.
Every snippet must match exactly once, otherwise nothing is written.
"""
from pathlib import Path

DWELL_METHOD = '''    def _dwell_keep_mask(self, path, count):
        """M2: True = use this frame as a training anchor.

        A frame is static when the fastest joint moves slower than
        dwell_speed_deg (deg/s) and the commanded gripper changes less than
        dwell_gripper_m since the previous frame.  In each static run inside
        one step keep the first dwell_keep[step] (or dwell_keep["default"])
        frames and the last dwell_tail frames; drop the middle.
        """
        traj = self._read_json(path)
        q = np.asarray(get_value(traj, "proprios.arm_joint"), dtype=np.float64)
        g = np.asarray(get_value(traj, "actions.gripper_pos"), dtype=np.float64).reshape(len(q), -1)[:, 0]
        source = traj.get("source") or {}
        steps = list(source.get("step_index") or [0] * count)
        fps = float(traj.get("fps", 10))
        speed = np.r_[np.inf, np.abs(np.diff(q, axis=0)).max(1) * (180.0 / np.pi) * fps]
        grip = np.r_[np.inf, np.abs(np.diff(g))]
        static = (speed < self.dwell_speed_deg) & (grip < self.dwell_gripper_m)
        keep = np.ones(count, dtype=bool)
        i = 0
        while i < count:
            if not static[i]:
                i += 1
                continue
            j = i
            while j < count and static[j] and steps[j] == steps[i]:
                j += 1
            head = int(self.dwell_keep.get(str(steps[i]), self.dwell_keep.get("default", 10**9)))
            if j - i > head + self.dwell_tail:
                keep[i + head:j - self.dwell_tail] = False
                key = f"{source.get('dataset', '?')}/step{steps[i]}"
                self.dwell_removed[key] = self.dwell_removed.get(key, 0) + (j - i - head - self.dwell_tail)
            i = j
        return keep

'''

EDITS = {
    "mibot/data/datasets/json_dataset.py": [
        (
            '        self.step_prompt = bool(data.get("step_prompt", params.get("step_prompt", False)))\n',
            '        self.step_prompt = bool(data.get("step_prompt", params.get("step_prompt", False)))\n'
            '        # M2: gripper target = recorded command (absolute), not command - current opening.\n'
            '        self.gripper_absolute = bool(data.get("gripper_absolute", False))\n'
            '        # M2: thin long static runs as training anchors (None = off).\n'
            '        dwell = data.get("dwell_keep")\n'
            '        self.dwell_keep = {str(k): int(v) for k, v in dict(dwell).items()} if dwell else None\n'
            '        self.dwell_tail = int(data.get("dwell_tail", 10))\n'
            '        self.dwell_speed_deg = float(data.get("dwell_speed_deg", 2.0))\n'
            '        self.dwell_gripper_m = float(data.get("dwell_gripper_m", 0.001))\n'
            '        self.dwell_removed = {}\n',
        ),
        (
            '            source_samples.extend({"file": path, "frame_index": i} for i in range(count))\n',
            '            keep = self._dwell_keep_mask(path, count) if self.dwell_keep else None\n'
            '            source_samples.extend(\n'
            '                {"file": path, "frame_index": i} for i in range(count) if keep is None or keep[i]\n'
            '            )\n',
        ),
        (
            '    def _prompt(self, traj, frame=None):\n',
            DWELL_METHOD + '    def _prompt(self, traj, frame=None):\n',
        ),
        (
            '                self._delta(traj, "proprios.gripper_pos", "actions.gripper_pos", frame, steps),\n'
            '            )\n'
            '            joint, gripper = normalize_action_parts(values, self.mean, self.std, self.action_layout)\n',
            '                self._pad(self._future(traj, "actions.gripper_pos", frame, steps), steps)\n'
            '                if self.gripper_absolute\n'
            '                else self._delta(traj, "proprios.gripper_pos", "actions.gripper_pos", frame, steps),\n'
            '            )\n'
            '            joint, gripper = normalize_action_parts(values, self.mean, self.std, self.action_layout)\n',
        ),
    ],
    "scripts/serve_openpi_policy.py": [
        (
            '        self.state_valid = self.q99 > self.q01\n',
            '        self.state_valid = self.q99 > self.q01\n'
            '        # M2: gripper predicted as an absolute target (see JsonDataset.gripper_absolute).\n'
            '        self.gripper_absolute = bool(data_cfg.get("gripper_absolute", False))\n',
        ),
        (
            '            self.metadata["gripper_range"] = [0.0, 0.1]\n',
            '            self.metadata["gripper_range"] = [0.0, 0.1]\n'
            '            self.metadata["gripper_mode"] = "absolute" if self.gripper_absolute else "delta"\n',
        ),
        (
            '        result = {"actions": actions, "prompt_used": getattr(self, "last_prompt", None)}\n',
            '        if getattr(self, "gripper_absolute", False) and getattr(self, "action_layout", "") == "nero_joint":\n'
            '            actions[:, 7] = delta[:, 7]\n'
            '        result = {"actions": actions, "prompt_used": getattr(self, "last_prompt", None)}\n',
        ),
    ],
    "scripts/panhe/train_nero.sh": [
        (
            'if [[ ${XR1_CHECK_ONLY:-0} == 1 ]]; then\n'
            '  exec bash scripts/train_910c_full_joint.sh auto "$EXP" "$@"\n'
            'fi\n',
            '# Fixed step budget (M2: dwell thinning makes the JSON frame count != samples):\n'
            '#   XR1_MAX_STEPS=<n> keeps finite epochs but uses n for max_steps and the LR schedule.\n'
            'STEPS=auto; EXTRA=()\n'
            'if [[ -n ${XR1_MAX_STEPS:-} ]]; then\n'
            '  STEPS=$XR1_MAX_STEPS; EXTRA=("+data.params.finite_epoch=true" "+trainer.max_epochs=$XR1_EPOCHS"); unset XR1_EPOCHS\n'
            'fi\n'
            'if [[ ${XR1_CHECK_ONLY:-0} == 1 ]]; then\n'
            '  exec bash scripts/train_910c_full_joint.sh "$STEPS" "$EXP" "${EXTRA[@]}" "$@"\n'
            'fi\n',
        ),
        (
            'bash scripts/train_910c_full_joint.sh auto "$EXP" "$@" 2>&1 | tee "$XR1_OUTPUT_ROOT.log"\n',
            'bash scripts/train_910c_full_joint.sh "$STEPS" "$EXP" "${EXTRA[@]}" "$@" 2>&1 | tee "$XR1_OUTPUT_ROOT.log"\n',
        ),
    ],
}

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
    Path(path).write_text(text)
    print("edited", path)
