"""Write configs/data/nero3_m2_20261106.yaml from the baseline data config.

Changes vs baseline: + failure-prefix datasets; gripper column (7) of the
action mean/std set to 0.05/0.05 so the absolute 0..0.1 target maps to [-1, 1];
gripper_absolute and dwell thinning switched on.  Joint stats and state
quantiles stay identical to the baseline so M2 differs from M1 only where intended.
"""
from omegaconf import OmegaConf

BASE = "configs/data/nero3_single_20261031.yaml"
OUT = "configs/data/nero3_m2_20261106.yaml"
FAIL = "/mnt/panhe/dataset/xr1_nero_fail_prefix_20261106/data"

cfg = OmegaConf.load(BASE)
td = cfg.data.params.train_datasets
td.paths = list(td.paths) + [f"{FAIL}/{n}" for n in ("nero_weigh_fail_v1", "nero_temp_fail_v1", "nero_wind_fail_v1")]
mean, std = OmegaConf.to_container(td.mean), OmegaConf.to_container(td.std)
for row in mean:
    row[7] = 0.05
for row in std:
    row[7] = 0.05
td.mean, td.std = mean, std
td.gripper_absolute = True
td.dwell_keep = {"default": 10, "3": 20}
td.dwell_tail = 10
td.dwell_speed_deg = 2.0
td.dwell_gripper_m = 0.001
with open(OUT, "w") as f:
    f.write("# @package _global_\n# M2: baseline + failure prefixes, absolute gripper, dwell thinning (see scripts/panhe/make_m2_config.py)\n")
    f.write(OmegaConf.to_yaml(cfg))
print("wrote", OUT, "paths:", len(td.paths))
