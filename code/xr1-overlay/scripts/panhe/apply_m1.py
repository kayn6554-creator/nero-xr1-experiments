"""Apply the M1 edits (step text in the prompt) to /mnt/panhe/xr1 in place.

Each edit replaces an exact snippet that must occur exactly once; otherwise the
script stops without writing anything.  Run from the xr1 directory.
"""
from pathlib import Path

EDITS = {
    "mibot/data/datasets/json_dataset.py": [
        (
            '        self.finite_epoch = bool(params.get("finite_epoch", False))\n',
            '        self.finite_epoch = bool(params.get("finite_epoch", False))\n'
            '        # M1: append "Step k/4: ..." for the current frame (off = original behaviour).\n'
            '        self.step_prompt = bool(data.get("step_prompt", params.get("step_prompt", False)))\n',
        ),
        ("        prompt = self._prompt(traj)\n", "        prompt = self._prompt(traj, frame)\n"),
        ("    def _prompt(self, traj):\n", "    def _prompt(self, traj, frame=None):\n"),
        (
            '        if len(conversations) >= 2:\n            conversations[0]["value"] += " /no_cot"\n',
            '        if len(conversations) >= 2:\n'
            '            if getattr(self, "step_prompt", False):\n'
            '                conversations[0]["value"] += " " + step_text_for_frame(traj, frame)\n'
            '            conversations[0]["value"] += " /no_cot"\n',
        ),
        (
            "logger = logging.get_logger(__name__)\n",
            "from mibot.utils.nero_steps import step_text_for_frame\n\nlogger = logging.get_logger(__name__)\n",
        ),
    ],
    "scripts/train_910c_full_joint.sh": [
        (
            "XR1_DIR=/mnt/wzh/XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1\n",
            "XR1_DIR=${XR1_DIR:-/mnt/wzh/XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1}\n",
        ),
    ],
    "scripts/serve_openpi_policy.py": [
        (
            "        prompt = obs.get(\"prompt\") or self.default_prompt\n",
            "        prompt = obs.get(\"prompt\") or self.default_prompt\n"
            "        self.last_prompt = prompt.strip() if isinstance(prompt, str) else None\n",
        ),
        (
            '        result = {"actions": actions}\n',
            '        result = {"actions": actions, "prompt_used": getattr(self, "last_prompt", None)}\n',
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
