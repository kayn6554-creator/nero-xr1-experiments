#!/usr/bin/env python3
"""Pure logic tests for the 2026-11-16 M4/EF4/M5 serving patch.

Run in the 113 XR1 Python environment:
  PYTHONPATH=/data/hep/panhe/xr1:/path/to/patch_bundle/113/scripts \
    python tests/test_step_flag_logic.py
"""
import numpy as np
from serve_openpi_policy import apply_step_flag_boundary, merge23_prompt
from mibot.utils.io import build_action_mask

for prompt, expected in [
    ("Task Step 1/4: grasp the tool.", "Task Step 1/3: grasp the tool."),
    ("Task Step 2/4: move above the target.", "Task Step 2/3: move to the target and measure."),
    ("Task Step 3/4: insert and measure.", "Task Step 2/3: move to the target and measure."),
    ("Task Step 4/4: return the tool.", "Task Step 3/3: return the tool."),
]:
    assert merge23_prompt(prompt) == expected
assert merge23_prompt("Task without step suffix") == "Task without step suffix"

mask = build_action_mask(30, action_layout="nero_joint_sb")
assert mask.shape == (30, 60)
assert np.all(mask[:, :9] == 1) and np.all(mask[:, 9:] == 0)

state = np.arange(8, dtype=np.float32)
actions = np.stack([state + i + 1 for i in range(5)]).astype(np.float32)

out, info = apply_step_flag_boundary(actions, state, np.array([0, 0.2, 0.1, 0, 0]))
assert not info["detected"] and info["boundary_index"] is None
assert np.array_equal(out, actions)

out, info = apply_step_flag_boundary(actions, state, np.array([0, 1, 0, 0, 0]))
assert not info["detected"]

out, info = apply_step_flag_boundary(actions, state, np.array([0, 0, 1, 1, 1]))
assert info["detected"] and info["boundary_index"] == 2 and info["trigger_index"] == 3
assert np.array_equal(out[:2], actions[:2])
assert np.all(out[2:] == actions[1])

out, info = apply_step_flag_boundary(actions, state, np.array([1, 1, 1, 1, 1]))
assert info["boundary_index"] == 0
assert np.all(out == state)

print("M4/EF4 merge23 + M5 action mask/Step Flag tests: PASS")
