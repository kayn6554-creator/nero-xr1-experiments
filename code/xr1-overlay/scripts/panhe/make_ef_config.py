"""Write configs/data/nero3_ee_20261110.yaml for the EF model and self-test it.

Same data, dwell thinning, absolute gripper and state q01/q99 as M3
(nero3_m3_20261110.yaml); only action_layout -> nero_ee and the action mean/std:
  slots 0-5, 7: per-timestep mean/std of the nero_ee targets over every chunk
                start of every episode in the M3 paths (no padding frames);
  slot 6 (gripper, absolute 0..0.1): 0.05 / 0.05 as in M2/M3;
  other slots: 0 / 1 (masked).
Self-test (needs apply_ef.py applied): builds the real JsonDataset, takes random
samples, un-normalizes the action, rebuilds absolute flange poses + j3, checks
them against FK of the recorded joints, then runs our IK (seed = state, j3 target)
and checks it gives back the recorded joints.  Prints XR1_MAX_STEPS.

Run from /mnt/panhe/xr1 after `source /usr/local/Ascend/cann-9.0.0/set_env.sh`:
  .venv/bin/python scripts/panhe/make_ef_config.py
"""
import glob
import math
import os
import random

import numpy as np
from omegaconf import OmegaConf

from mibot.utils import nero_kin as K

BASE = "configs/data/nero3_m3_20261110.yaml"
OUT = "configs/data/nero3_ee_20261110.yaml"
H, EPOCHS, CARDS = 30, 3, 8

assert not os.path.exists(OUT), f"{OUT} already exists, not overwriting"
cfg = OmegaConf.load(BASE)
td = cfg.data.params.train_datasets
files = sorted(f for p in td.paths for f in glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
print("episodes:", len(files))


def fk_all(q):
    T = np.stack([K.fk(x) for x in q])
    return T[:, :3, :3], T[:, :3, 3]


S = np.zeros((H, 7)); SS = np.zeros((H, 7)); N = np.zeros(H)
import json  # noqa: E402
for f in files:
    traj = json.load(open(f))
    qs = np.asarray(traj["proprios"]["arm_joint"], dtype=np.float64)
    qa = np.asarray(traj["actions"]["arm_joint"], dtype=np.float64)
    Rs, ps = fk_all(qs)
    Ra, pa = fk_all(qa)
    n = len(qs)
    for k in range(min(H, n)):
        m = n - k
        R0t = np.transpose(Rs[:m], (0, 2, 1))
        pos = np.einsum("nij,nj->ni", R0t, pa[k:] - ps[:m])
        aa = K.rot_log_batch(R0t @ Ra[k:])
        elbow = qa[k:, 2:3] - qs[:m, 2:3]
        v = np.concatenate([pos, aa, elbow], 1)
        S[k] += v.sum(0); SS[k] += (v * v).sum(0); N[k] += m
mu = S / N[:, None]
sd = np.sqrt(np.maximum(SS / N[:, None] - mu ** 2, 0))
sd = np.where(sd > 1e-6, sd, 1.0)
mean, std = np.zeros((H, 60)), np.ones((H, 60))
mean[:, 0:6], std[:, 0:6] = mu[:, :6], sd[:, :6]
mean[:, 7], std[:, 7] = mu[:, 6], sd[:, 6]
mean[:, 6], std[:, 6] = 0.05, 0.05
print("std at k=0 :", np.round(std[0, :8], 5).tolist())
print("std at k=29:", np.round(std[29, :8], 5).tolist())

td.action_layout = "nero_ee"
td.mean, td.std = mean.tolist(), std.tolist()
with open(OUT, "w") as fh:
    fh.write("# @package _global_\n# EF: M3 data + action_layout nero_ee (flange pose delta, gripper, j3 delta); "
             "stats from scripts/panhe/make_ef_config.py\n")
    fh.write(OmegaConf.to_yaml(cfg))
print("wrote", OUT)

# ---------------------------------------------------------------- self-test on the real dataset class
from mibot.data.datasets.json_dataset import JsonDataset  # noqa: E402

params = OmegaConf.to_container(OmegaConf.load(OUT).data.params, resolve=False)
params["finite_epoch"] = True
params["max_steps"] = 1
ds = JsonDataset(params)
rng = random.Random(0)
pos_err, rot_err, j_err, z = [], [], [], []
for idx in rng.sample(range(len(ds)), 20):
    item = ds[idx]
    s = ds.samples[idx]
    traj = ds._read_json(s["file"])
    t, steps = s["frame_index"], int(item["vlm_action_actual_length"])
    a = item["action"].numpy().astype(np.float64) * (np.asarray(ds.std) + 1e-6) + np.asarray(ds.mean)
    z.append(np.abs(item["action"].numpy()[:steps, :8]).max())
    qs = np.asarray(traj["proprios"]["arm_joint"][t], dtype=np.float64)
    T0 = K.fk(qs)
    q_prev = qs.copy()
    for k in range(steps):
        q_rec = np.asarray(traj["actions"]["arm_joint"][t + k], dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = T0[:3, :3] @ K.rot_exp(a[k, 3:6])
        T[:3, 3] = T0[:3, 3] + T0[:3, :3] @ a[k, 0:3]
        T_rec = K.fk(q_rec)
        pos_err.append(1000 * np.linalg.norm(T[:3, 3] - T_rec[:3, 3]))
        rot_err.append(np.degrees(np.linalg.norm(K.rot_log(T[:3, :3] @ T_rec[:3, :3].T))))
        q, ok, _, _ = K.ik(T, q_prev, q3_target=qs[2] + a[k, 7])
        j_err.append(np.degrees(np.abs(q - q_rec).max()))
        q_prev = q
print(f"decode check (20 samples): pos err max {max(pos_err):.3f} mm, rot err max {max(rot_err):.4f} deg, "
      f"IK joints vs recorded max {max(j_err):.3f} deg, |normalized action| max {max(z):.1f}")
print("SELF-TEST", "PASS" if max(pos_err) < 0.5 and max(rot_err) < 0.05 and max(j_err) < 0.5 else "CHECK")
print(f"samples={len(ds)}  XR1_MAX_STEPS={EPOCHS * math.ceil(len(ds) / (int(td.batch_size) * CARDS))}")
