#!/usr/bin/env python3
"""EF check B (offline): if the model predicted end-effector poses perfectly,
could our IK turn them back into the recorded joint commands?

For sampled chunks of recorded XR-1 JSON episodes, exactly like deployment:
  seed = recorded follower state at the chunk start;
  for k in 0..29: target = FK(recorded action joints at t+k);
                  q_k = IK(target, seed = q_{k-1}, null-space pull toward q_{k-1}),
                  with --elbow j3 also j3 = recorded action j3 (the EF model's elbow dim).
Reports per dataset: IK residual (mm / deg), failures, deviation of the IK
joints from the recorded command joints (the 7-DoF elbow freedom), and IK
joint jumps beyond what the recording itself did.  Reads JSON only; numpy only.

Usage on Atlas (from /mnt/panhe/xr1, no NPU, no CANN needed):
  .venv/bin/python scripts/panhe/ef_ik_roundtrip.py [--elbow j3|none] [--episodes 20] [--stride 10]
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nero_kin as K  # noqa: E402

DIRS = [
    "/mnt/wzh/dataset/xr1_nero3_single_20261031/data/nero_weigh_v1",
    "/mnt/wzh/dataset/xr1_nero3_single_20261031/data/nero_temp_v1",
    "/mnt/wzh/dataset/xr1_nero3_single_20261031/data/nero_wind_v1",
    "/mnt/panhe/dataset/xr1_nero_new_20261110/data/nero_noise_v1",
    "/mnt/panhe/dataset/xr1_nero_new_20261110/data/nero_ircup_v1",
    "/mnt/panhe/dataset/xr1_nero_new_20261110/data/nero_tds_v1",
    "/mnt/panhe/dataset/xr1_nero_new_20261110/data/nero_ph_v1",
    "/mnt/panhe/dataset/xr1_nero_new_20261110/data/nero_stemp_v1",
]


def pct(x, p):
    return round(float(np.percentile(x, p)), 3) if len(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=DIRS)
    ap.add_argument("--episodes", type=int, default=20, help="random episodes per dataset (0 = all)")
    ap.add_argument("--stride", type=int, default=10, help="chunk start every N frames")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--elbow", choices=["none", "j3"], default="j3",
                    help="none: 6-D pose only; j3: pose + recorded j3 target (EF action with elbow dimension)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = random.Random(0)
    report = {}
    t0 = time.time()
    for d in a.dirs:
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        if not files:
            print(f"skip {d}: no JSON")
            continue
        if a.episodes:
            files = rng.sample(files, min(a.episodes, len(files)))
        perr, rerr, dev, dev_end, extra_jump, ik_jump = [], [], [], [], [], []
        fails = calls = limit_hits = 0
        for f in files:
            traj = json.load(open(f))
            qs = np.asarray(traj["proprios"]["arm_joint"], dtype=np.float64)
            qa = np.asarray(traj["actions"]["arm_joint"], dtype=np.float64)
            n = len(qa)
            for t in range(0, n, a.stride):
                q_prev = qs[t].copy()
                rec_prev = qs[t]
                last_dev = None
                for k in range(min(a.horizon, n - t)):
                    target = K.fk(qa[t + k])
                    q3 = qa[t + k][K.ELBOW_JOINT] if a.elbow == "j3" else None
                    q, ok, ep, er = K.ik(target, q_prev, q3_target=q3)
                    calls += 1
                    fails += not ok
                    limit_hits += bool(np.any((q <= K.LIMITS[:, 0] + 1e-6) | (q >= K.LIMITS[:, 1] - 1e-6)))
                    perr.append(1000 * ep)
                    rerr.append(np.degrees(er))
                    dv = np.degrees(np.abs(q - qa[t + k]))
                    dev.append(dv)
                    last_dev = dv.max()
                    jump = np.degrees(np.abs(q - q_prev).max())
                    rec_jump = np.degrees(np.abs(qa[t + k] - rec_prev).max())
                    ik_jump.append(jump)
                    extra_jump.append(jump - rec_jump)
                    q_prev, rec_prev = q, qa[t + k]
                if last_dev is not None:
                    dev_end.append(last_dev)
        dev = np.array(dev)
        name = os.path.basename(d.rstrip("/"))
        r = {
            "episodes": len(files), "ik_calls": calls,
            "fail_pct": round(100 * fails / max(calls, 1), 3),
            "joint_limit_hit_pct": round(100 * limit_hits / max(calls, 1), 3),
            "pos_err_mm": {"p50": pct(perr, 50), "p99": pct(perr, 99), "max": pct(perr, 100)},
            "rot_err_deg": {"p50": pct(rerr, 50), "p99": pct(rerr, 99), "max": pct(rerr, 100)},
            "dev_from_recorded_deg_per_joint_p95": [pct(dev[:, j], 95) for j in range(7)],
            "dev_from_recorded_deg_per_joint_max": [pct(dev[:, j], 100) for j in range(7)],
            "dev_at_chunk_end_deg": {"p50": pct(dev_end, 50), "p95": pct(dev_end, 95), "max": pct(dev_end, 100)},
            "ik_jump_deg": {"p99": pct(ik_jump, 99), "max": pct(ik_jump, 100)},
            "extra_jump_vs_recording_deg": {"p99": pct(extra_jump, 99), "max": pct(extra_jump, 100)},
        }
        report[name] = r
        print(name, json.dumps(r), flush=True)
    ok = all(r["fail_pct"] < 0.5 and r["pos_err_mm"]["p99"] < 1 and r["rot_err_deg"]["p99"] < 0.1
             and r["dev_at_chunk_end_deg"]["p95"] < 3 and r["extra_jump_vs_recording_deg"]["max"] < 3
             for r in report.values())
    print(f"time {time.time() - t0:.0f}s")
    print("RESULT:", "PASS (IK reproduces the recorded commands)" if ok else
          "CHECK numbers above (thresholds: fail<0.5%, pos p99<1mm, rot p99<0.1deg, "
          "chunk-end dev p95<3deg, extra jump max<3deg)")
    a.out = a.out or f"/mnt/panhe/output/ef_ik_roundtrip_1110_{a.elbow}.json"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
