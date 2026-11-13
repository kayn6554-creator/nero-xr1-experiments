"""M4 / EF4 (merge23): let JsonDataset use the merged step text (run from /mnt/panhe/xr1 on Atlas).

Adds the data option train_datasets.step_merge23 (default false = old behaviour, so M1-M3/EF
are unchanged).  With step_prompt=true and step_merge23=true, frames of steps 2 and 3 get the
same text "Step 2/3: move to the target and measure." (see mibot/utils/nero_steps.py).
Needs the 11-12 nero_steps.py (merge23_prompt / step_text(..., merge23)) in mibot/utils/.
Backup: mibot/data/datasets/json_dataset.py.before_m4
"""
import shutil
from pathlib import Path

PATH = "mibot/data/datasets/json_dataset.py"
EDITS = [
    ('        self.step_prompt = bool(data.get("step_prompt", params.get("step_prompt", False)))\n',
     '        self.step_prompt = bool(data.get("step_prompt", params.get("step_prompt", False)))\n'
     '        self.step_merge23 = bool(data.get("step_merge23", False))  # M4: steps 2+3 share one text\n'),
    ('                conversations[0]["value"] += " " + step_text_for_frame(traj, frame)\n',
     '                conversations[0]["value"] += " " + step_text_for_frame(\n'
     '                    traj, frame, getattr(self, "step_merge23", False))\n'),
]
from mibot.utils import nero_steps  # noqa: E402
if not hasattr(nero_steps, "merge23_prompt"):
    raise SystemExit("mibot/utils/nero_steps.py is the old one (no merge23_prompt); copy the 11-12 version first")
text = Path(PATH).read_text()
if "step_merge23" in text:
    raise SystemExit("already applied, nothing written")
for old, new in EDITS:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{PATH}: expected 1 match, found {n}:\n{old}")
    text = text.replace(old, new)
shutil.copy2(PATH, PATH + ".before_m4")
Path(PATH).write_text(text)
print("edited", PATH, "(backup .before_m4)")
print("check:", nero_steps.step_text("temp", 2, True), "|", nero_steps.step_text("temp", 3, True), "|",
      nero_steps.step_text("temp", 4, True), "| old:", nero_steps.step_text("temp", 3))
