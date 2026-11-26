"""Write configs/data/nero3_m3_20261110.yaml = M2 data config + the 11-10 datasets.

Everything else (mean/std, q01/q99, gripper_absolute, dwell thinning) is copied
unchanged from nero3_m2_20261106.yaml, so M3 differs from M2 only in the data.
Then builds the real JsonDataset (finite epoch, dwell thinning on) to count the
training samples and prints XR1_MAX_STEPS = epochs * ceil(samples / global batch),
the same way M2's 2793 = 3 * ceil(74441 / 80) was obtained.

Run from /mnt/panhe/xr1:  .venv/bin/python scripts/panhe/make_m3_config.py
"""
import math
import os

from omegaconf import OmegaConf

BASE = "configs/data/nero3_m2_20261106.yaml"
OUT = "configs/data/nero3_m3_20261110.yaml"
NEW = "/mnt/panhe/dataset/xr1_nero_new_20261110/data"
NAMES = ("nero_noise_v1", "nero_ircup_v1", "nero_tds_v1", "nero_ph_v1", "nero_stemp_v1")
EPOCHS, CARDS = 3, 8

for n in NAMES:
    assert os.path.isdir(f"{NEW}/{n}"), f"missing {NEW}/{n} (run prepare_nero_new_1110.py first)"
assert not os.path.exists(OUT), f"{OUT} already exists, not overwriting"

cfg = OmegaConf.load(BASE)
td = cfg.data.params.train_datasets
td.paths = list(td.paths) + [f"{NEW}/{n}" for n in NAMES]
with open(OUT, "w") as f:
    f.write("# @package _global_\n# M3: M2 data config + 11-10 datasets (noise, ircup, tds, ph, stemp); "
            "stats unchanged (see scripts/panhe/make_m3_config.py)\n")
    f.write(OmegaConf.to_yaml(cfg))
print("wrote", OUT, "paths:", len(td.paths))

# Count samples exactly as training will see them.
from mibot.data.datasets.json_dataset import JsonDataset  # noqa: E402

params = OmegaConf.to_container(cfg.data.params, resolve=False)
params["finite_epoch"] = True
params["max_steps"] = 1
ds = JsonDataset(params)
batch = int(td.batch_size) * CARDS
per_dataset = {}
for s in ds.samples:
    key = s["file"].split("/")[-2]
    per_dataset[key] = per_dataset.get(key, 0) + 1
for k in sorted(per_dataset):
    print(f"  {k:24s} {per_dataset[k]:6d} samples")
print("dwell frames removed:", sum(ds.dwell_removed.values()))
print(f"samples={len(ds)}  global_batch={batch}  epochs={EPOCHS}")
print(f"XR1_MAX_STEPS={EPOCHS * math.ceil(len(ds) / batch)}")