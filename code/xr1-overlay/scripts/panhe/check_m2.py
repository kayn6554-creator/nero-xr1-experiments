"""CPU check of the M2 data: sample counts, dwell thinning, gripper targets, one full sample."""
import math, os, sys
os.environ.setdefault("WORLD_SIZE", "1")
import numpy as np
from omegaconf import OmegaConf
from mibot.data.datasets.json_dataset import JsonDataset

name = sys.argv[1] if len(sys.argv) > 1 else "nero3_m2_20261106"
cfg = OmegaConf.load(f"configs/data/{name}.yaml")
params = OmegaConf.to_container(cfg.data.params, resolve=False)
params["max_steps"] = 1
params["finite_epoch"] = True
params["train_datasets"]["step_prompt"] = True
ds = JsonDataset(params)
total = 0
for f in ds.files:
    total += int(ds._read_json(f)["num_frames"])
print(f"files {len(ds.files)}  json frames {total}  training samples {len(ds)}  removed {total - len(ds)}")
for k, v in sorted(ds.dwell_removed.items()):
    print("  removed", k, v)
for world, acc in ((8, 1), (4, 2)):
    per_epoch = math.ceil(len(ds) / (world * 10 * acc))
    print(f"steps (world {world}, accumulation {acc}): {per_epoch}/epoch, 3 epochs = {3 * per_epoch}")
rng = np.random.default_rng(0)
g = []
for idx in rng.choice(len(ds), 200, replace=False):
    s = ds.samples[idx]; traj = ds._read_json(s["file"]); fr = s["frame_index"]
    steps = min(ds.action_length, int(traj["num_frames"]) - fr)
    v = ds._pad(ds._future(traj, "actions.gripper_pos", fr, steps), steps)
    g.append((v[:, 0] - ds.mean[:, 7]) / (ds.std[:, 7] + 1e-6))
g = np.concatenate(g)
print(f"normalized gripper target: min {g.min():.3f} max {g.max():.3f} (expect within about [-1, 1])")
fail = next(i for i, s in enumerate(ds.samples) if "_fail_" in s["file"])
item = ds[fail]
print("full sample from", ds.samples[fail]["file"].split("/data/")[-1], "frame", ds.samples[fail]["frame_index"],
      "action", tuple(item["action"].shape), "gripper[0]", float(item["action"][0, 7]))
print("prompt:", item["messages"][0]["content"][-1]["text"][-120:])
