"""CPU check of M1 prompts: print the prompt at each step boundary of a few episodes."""
import os, sys
os.environ.setdefault("WORLD_SIZE", "1")
from omegaconf import OmegaConf
from mibot.data.datasets.json_dataset import JsonDataset

cfg = OmegaConf.load("configs/data/nero3_single_20261031.yaml")
params = OmegaConf.to_container(cfg.data.params, resolve=False)
params["max_steps"] = 1
params["finite_epoch"] = True
on = sys.argv[1:] != ["off"]
params["train_datasets"]["step_prompt"] = on
ds = JsonDataset(params)
print("step_prompt =", on, "files =", len(ds.files))
for name in ("nero_weigh_v1", "nero_temp_v1", "nero_wind_v1"):
    path = next(f for f in ds.files if f"/{name}/" in f)
    traj = ds._read_json(path)
    steps = traj["source"]["step_index"]
    frames = [0] + [i for i in range(1, len(steps)) if steps[i] != steps[i - 1]]
    for frame in frames:
        text = ds._prompt(traj, frame)["conversations"][0]["value"]
        print(f"{name} frame {frame:4d} step {steps[frame]}: {text!r}")
