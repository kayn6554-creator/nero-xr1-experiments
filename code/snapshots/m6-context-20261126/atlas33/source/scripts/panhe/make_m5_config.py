"""Write configs/data/nero3_m5sb_20261112.yaml = M3 data config with action_layout nero_joint_sb
and slot-8 (step flag) mean/std = 0.5 / 0.5; everything else identical to M3.  Then self-test
on the real JsonDataset: slot 8 present, in the action mask, flag rate per step.

Run from /mnt/panhe/xr1 after `source /usr/local/Ascend/cann-9.0.0/set_env.sh`:
  .venv/bin/python scripts/panhe/make_m5_config.py
"""
import math
import os
import random

import numpy as np
from omegaconf import OmegaConf

BASE = "configs/data/nero3_m3_20261110.yaml"
OUT = "configs/data/nero3_m5sb_20261112.yaml"
assert not os.path.exists(OUT), f"{OUT} already exists, not overwriting"
cfg = OmegaConf.load(BASE)
td = cfg.data.params.train_datasets
mean, std = np.asarray(td.mean, dtype=np.float64), np.asarray(td.std, dtype=np.float64)
assert mean.shape == (30, 60) and np.all(mean[:, 8] == 0) and np.all(std[:, 8] == 1), "slot 8 not unused in M3 stats"
mean[:, 8], std[:, 8] = 0.5, 0.5
td.mean, td.std = mean.tolist(), std.tolist()
td.action_layout = "nero_joint_sb"
with open(OUT, "w") as fh:
    fh.write("# @package _global_\n# M5: M3 data + action_layout nero_joint_sb (slot 8 = next-step flag, mean/std 0.5)\n")
    fh.write(OmegaConf.to_yaml(cfg))
print("wrote", OUT)

from mibot.data.datasets.json_dataset import JsonDataset  # noqa: E402

params = OmegaConf.to_container(OmegaConf.load(OUT).data.params, resolve=False)
params["finite_epoch"] = True
params["max_steps"] = 1
ds = JsonDataset(params)
rng = random.Random(0)
flags, on = [], 0
for i in rng.sample(range(len(ds)), 200):
    item = ds[i]
    a, m = item["action"].numpy(), item["action_mask"].numpy()
    assert m[:, 8].max() == 1, "slot 8 not in the action mask"
    f = a[:, 8] * 0.5 + 0.5
    assert np.all((np.abs(f) < 1e-3) | (np.abs(f - 1) < 1e-3)), f
    assert np.all(np.diff(f) >= -1e-3), "flag must not go 1 -> 0 inside a chunk"
    s = ds.samples[i]
    traj = ds._read_json(s["file"])
    idx = traj["source"]["step_index"]
    t = s["frame_index"]
    exp = [float(idx[min(t + k, len(idx) - 1)] != idx[t]) for k in range(30)]
    assert np.allclose(f, exp, atol=1e-3), (f, exp)
    on += f.max() > 0.5
    # joints / gripper identical to plain nero_joint
    assert np.abs(a[:, 9:]).max() == 0
print(f"SELF-TEST PASS: 200 samples, {on} contain a step change ({on / 2:.0f}%)")
print(f"samples={len(ds)}  XR1_MAX_STEPS={3 * math.ceil(len(ds) / 80)}")