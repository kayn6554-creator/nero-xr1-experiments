"""NERO step text, shared by training (JsonDataset) and the robot client.

Training and inference must build exactly the same instruction string:

    "<task> Step k/4: <step name>."

``k`` is the frame's ``step_index`` (1..4).  During collection the step key is
pressed at the END of the previous step, so ``step_index`` switches to k+1 at
that moment; at test time the operator presses "step done" at the same moment.

11-12 merge23 (M4 / EF4): steps 2 and 3 share one text, because a 30-step chunk
usually runs from step 2 straight into step 3 and the hand-pressed step key can
not follow it ("step skipping").  The recorded step_index is unchanged:
    step 1 -> "Step 1/3: <step 1 name>."
    step 2, 3 -> "Step 2/3: move to the target and measure."
    step 4 -> "Step 3/3: <step 4 name>."
The client keeps sending the normal 4-step prompt; a merge23 server converts it
with merge23_prompt(), so step key timing between 2 and 3 no longer matters.
"""
from __future__ import annotations

import re

MERGED_MIDDLE = "move to the target and measure"

STEPS = {
    "weigh": (
        "grasp the object",
        "move above the scale",
        "place the object on the scale",
        "move away and return",
    ),
    "temp": (
        "grasp the thermometer",
        "move above the container",
        "insert the thermometer into the water and measure",
        "put the thermometer back",
    ),
    "wind": (
        "grasp the anemometer",
        "approach the measuring point",
        "measure the wind speed",
        "put the anemometer back",
    ),
    # 11-09 新任务。键名要和数据集名 nero_<键>_v1 的前缀一致（task_key 按前缀找）
    "noise": (
        "grasp the noise meter",
        "move to the measuring point",
        "measure the noise",
        "put the noise meter back",
    ),
    "soil": (
        "grasp the soil probe",
        "move above the soil",
        "insert the probe into the soil and measure",
        "pull out the probe and put it back",
    ),
    "ircup": (     # 11-10：原 irwater（测水面）改成测容器外壁，数据集 nero_irwater_v1 改名 nero_ircup_v1
        "grasp the infrared thermometer",
        "approach the container",
        "aim at the container and measure",
        "put the infrared thermometer back",
    ),
    "irobj": (
        "grasp the infrared thermometer",
        "approach the object",
        "aim at the object and measure",
        "put the infrared thermometer back",
    ),
    # 11-10 新任务。键名不能是别的键的前缀加字母（task_key 按 nero_<键> 前缀找），所以土壤温度用 stemp 不用 soiltemp
    "tds": (
        "grasp the TDS pen",
        "move above the container",
        "insert the TDS pen into the water and measure",
        "put the TDS pen back",
    ),
    "ph": (
        "grasp the pH meter",
        "move above the container",
        "insert the pH meter into the water and measure",
        "put the pH meter back",
    ),
    "stemp": (
        "grasp the soil thermometer",
        "move above the soil",
        "insert the thermometer into the soil and measure",
        "pull out the thermometer and put it back",
    ),
}

# Client task keys (nero_tasks.json) -> step table keys.
CLIENT_TASKS = {"weigh": "weigh", "task2": "temp", "task3": "wind",
                "noise": "noise", "soil": "soil", "ircup": "ircup", "irobj": "irobj",
                "tds": "tds", "ph": "ph", "stemp": "stemp"}


def task_key(dataset: str) -> str:
    """Map a dataset name such as ``nero_temp_v1`` or ``nero_weigh_fail_v1``."""
    for key in STEPS:
        if str(dataset).startswith(f"nero_{key}"):
            return key
    raise ValueError(f"no NERO step table for dataset {dataset!r}")


def step_text(key: str, step: int, merge23: bool = False) -> str:
    names = STEPS[key]
    step = int(step)
    if not 1 <= step <= len(names):
        raise ValueError(f"step must be 1..{len(names)}, got {step}")
    if merge23:
        if len(names) != 4:
            raise ValueError("merge23 needs a 4-step table")
        k, name = {1: (1, names[0]), 2: (2, MERGED_MIDDLE), 3: (2, MERGED_MIDDLE), 4: (3, names[3])}[step]
        return f"Step {k}/3: {name}."
    return f"Step {step}/{len(names)}: {names[step - 1]}."


_STEP4 = re.compile(r"Step ([1-4])/4: (.+)\.$")


def merge23_prompt(prompt: str) -> str:
    """Convert a client prompt '<task> Step k/4: <name>.' to the merge23 text.
    Prompts without a step suffix are returned unchanged."""
    prompt = prompt.strip()
    m = _STEP4.search(prompt)
    if not m:
        return prompt
    k, name = int(m.group(1)), m.group(2)
    new = {1: f"Step 1/3: {name}.", 2: f"Step 2/3: {MERGED_MIDDLE}.",
           3: f"Step 2/3: {MERGED_MIDDLE}.", 4: f"Step 3/3: {name}."}[k]
    return prompt[:m.start()] + new


def step_prompt(task: str, key: str, step: int) -> str:
    """Full instruction sent by the client, e.g. for the OpenPI ``prompt`` field."""
    return f"{task.strip()} {step_text(key, step)}"


def step_text_for_frame(traj: dict, frame: int, merge23: bool = False) -> str:
    source = traj.get("source") or {}
    steps, dataset = source.get("step_index"), source.get("dataset")
    if steps is None or dataset is None:
        raise ValueError(
            f"step_prompt needs source.step_index and source.dataset in {traj.get('time', '<unknown>')}"
        )
    return step_text(task_key(dataset), steps[frame], merge23)
