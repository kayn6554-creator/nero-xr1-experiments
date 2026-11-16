#!/usr/bin/env python3
"""Client Step Flag parsing and manual step-file gate tests.

Run with the patch collector directory on PYTHONPATH:
  PYTHONPATH=collector python tests/test_client_stepflag.py
"""
import tempfile
from pathlib import Path
import numpy as np
import nero_vla_client as C

resp = {
    "actions": np.zeros((4, 8), dtype=np.float32),
    "step_flag": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32),
    "step_boundary": {
        "enabled": True, "detected": True, "boundary_index": 2,
        "trigger_index": 3, "threshold": 0.5, "consecutive_frames": 2,
        "max_value": 1.0, "first_value": -1.0,
    },
}
values, boundary = C.parse_step_flag(resp, 4)
assert values.shape == (4,) and boundary["boundary_index"] == 2

resp_no_flag = {"actions": np.zeros((4, 8), dtype=np.float32)}
values, boundary = C.parse_step_flag(resp_no_flag, 4)
assert values is None and boundary is None

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "step"
    p.write_text("2\n")
    assert C.wait_for_step_change(str(p), 1, C.time.monotonic() + 0.5) == 2

print("Client Step Flag tests: PASS (3 cases)")
