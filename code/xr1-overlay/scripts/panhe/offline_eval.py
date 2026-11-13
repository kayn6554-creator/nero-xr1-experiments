"""Offline action check before a model goes on the real robot.

For a few demo episodes per task, feed the recorded images + state (+ step text
when the model was trained with it) through the SAME inference wrapper the robot
uses (scripts/serve_openpi_policy.XR1OpenPIPolicy) and compare the predicted
chunk with the recorded commands.  Metrics per task and step:
  joint_deg     mean |pred - recorded| over the 30-step chunk (worst joint per step), degrees
  first10_deg   same for the first 10 steps (what E1 executes)
  grip_mm       mean |pred - recorded| gripper, mm
  close_dt      |first step pred gripper < 30 mm  -  first step recorded < 30 mm|, steps,
                only for chunks where the recording closes
These are training episodes (no held-out split), so this checks "trained correctly",
not generalisation.  Compare models against M0 on the same frames; a model much
worse than M0 should not go on the robot before we know why.

Usage (Atlas, one chip):
  XR1_DEVICE_TYPE=npu python scripts/panhe/offline_eval.py --model-dir <dir with config.py> \
      --device npu:0 --out /mnt/panhe/output/eval_<name>.json
"""
import argparse, glob, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch

XR1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(XR1))
from scripts.serve_openpi_policy import XR1OpenPIPolicy  # noqa: E402
from mibot.data.datasets.json_dataset import VideoReader  # noqa: E402
from mibot.utils.nero_steps import step_prompt, task_key  # noqa: E402

DATA = "/mnt/wzh/dataset/xr1_nero3_single_20261031/data"
NAMES = ("nero_weigh_v1", "nero_temp_v1", "nero_wind_v1")


def pick_episodes(n, seed=0):
    out = []
    for name in NAMES:
        files = sorted(glob.glob(f"{DATA}/{name}/episode_*.json"))
        out += random.Random(seed).sample(files, n)
    return out


def frames_of(traj, cam, idx):
    info = traj["images"][cam][0]
    batch = VideoReader(info["path"]).get_batch([i + int(info.get("start", 0)) for i in idx]).asnumpy()
    return list(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint", default="", help="default: the last *.ckpt in model-dir")
    ap.add_argument("--processor", default="/mnt/wzh/checkpoints/Qwen3-VL-4B-Instruct")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--episodes", type=int, default=5, help="per task")
    ap.add_argument("--stride", type=int, default=10, help="every N-th frame")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ckpt = a.checkpoint or sorted(Path(a.model_dir).glob("*.ckpt"))[-1].name
    t0 = time.time()
    policy = XR1OpenPIPolicy(a.model_dir, a.device, a.processor, None, checkpoint=ckpt)
    td = policy.cfg.data.params.train_datasets
    use_step = bool(td.get("step_prompt", False))
    print(f"loaded {a.model_dir} / {ckpt} in {time.time()-t0:.0f}s  step_prompt={use_step}  "
          f"gripper_mode={policy.metadata.get('gripper_mode', 'delta')}", flush=True)
    rows = []
    for path in pick_episodes(a.episodes):
        traj = json.load(open(path))
        n, src = traj["num_frames"], traj["source"]
        idx = list(range(0, n, a.stride))
        front, wrist = frames_of(traj, "front", idx), frames_of(traj, "wrist", idx)
        q = np.asarray(traj["proprios"]["arm_joint"]); g = np.asarray(traj["proprios"]["gripper_pos"])
        aq = np.asarray(traj["actions"]["arm_joint"]); ag = np.asarray(traj["actions"]["gripper_pos"])
        task = traj["instruction"]["general"][0]["conversations"][0]["value"].split("\n")[-1]
        for j, f in enumerate(idx):
            step = int(src["step_index"][f])
            prompt = step_prompt(task, task_key(src["dataset"]), step) if use_step else task
            torch.manual_seed(1000 + f)   # same noise for every model on the same frame
            out = policy.infer({"state": np.r_[q[f], g[f]].astype(np.float32),
                                "images": {"front": front[j], "wrist": wrist[j]}, "prompt": prompt})
            pred = np.asarray(out["actions"], dtype=np.float64)
            T = min(len(pred), n - f)
            gt = np.c_[aq[f:f + T], ag[f:f + T]]
            jerr = np.degrees(np.abs(pred[:T, :7] - gt[:, :7]).max(1))
            gerr = np.abs(pred[:T, 7] - gt[:, 7]) * 1000
            close_dt = None
            gt_close = np.where(gt[:, 7] < 0.03)[0]
            if len(gt_close) and g[f, 0] >= 0.03:
                pc = np.where(pred[:T, 7] < 0.03)[0]
                close_dt = abs(int(pc[0]) - int(gt_close[0])) if len(pc) else T
            rows.append({"dataset": src["dataset"], "episode": src["episode_index"], "frame": f, "step": step,
                         "joint_deg": float(jerr.mean()), "first10_deg": float(jerr[:10].mean()),
                         "grip_mm": float(gerr.mean()), "close_dt": close_dt})
        print(f"  {Path(path).parent.name}/{Path(path).name}: {len(idx)} frames", flush=True)
    summary = {}
    for name in NAMES:
        for step in (1, 2, 3, 4, "all"):
            r = [x for x in rows if x["dataset"] == name and (step == "all" or x["step"] == step)]
            if not r:
                continue
            cd = [x["close_dt"] for x in r if x["close_dt"] is not None]
            summary[f"{name}/step{step}"] = {
                "n": len(r), "joint_deg": round(float(np.mean([x["joint_deg"] for x in r])), 2),
                "first10_deg": round(float(np.mean([x["first10_deg"] for x in r])), 2),
                "grip_mm": round(float(np.mean([x["grip_mm"] for x in r])), 2),
                "close_dt": round(float(np.mean(cd)), 1) if cd else None, "close_n": len(cd)}
    Path(a.out).write_text(json.dumps({"model_dir": a.model_dir, "checkpoint": ckpt, "step_prompt": use_step,
                                       "summary": summary, "rows": rows}, indent=1))
    print(f"{'':28s}{'n':>5s}{'joint°':>8s}{'first10°':>9s}{'grip mm':>9s}{'close Δ':>8s}")
    for k, v in summary.items():
        print(f"{k:28s}{v['n']:5d}{v['joint_deg']:8.2f}{v['first10_deg']:9.2f}{v['grip_mm']:9.2f}{str(v['close_dt']):>8s}")
    print(f"done in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
