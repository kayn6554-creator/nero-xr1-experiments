"""Build Piper+NERO symlink index, mixed nero_ee_sb stats, and data YAML (pure NumPy)."""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
import numpy as np

H = 30
DIM = 60
NERO_REPEAT = 2
NERO_BASE_CFG = Path("configs/data/nero3_ee_20261110.yaml")
PIPER_DIR = Path("/mnt/wzh/dataset/xr1_piper_left_v2_20261110/data")
MIX_ROOT = Path("/mnt/panhe/dataset/xr1_ee_sb_mix_20261125")
OUT_YAML = Path("configs/data/nero3_piper_ee_sb_20261125.yaml")

spec = importlib.util.spec_from_file_location("ef_align_s", "mibot/utils/ef_align.py")
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)
NK = ea._NERO_KIN
PK = ea._PIPER_KIN


def nero_fk_batch(qs: np.ndarray) -> np.ndarray:
    n = len(qs)
    T = np.broadcast_to(np.eye(4, dtype=np.float64), (n, 4, 4)).copy()
    for idx, (d, a, alpha, off) in enumerate(NK.MDH):
        th = qs[:, idx] + off
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(th), np.sin(th)
        L = np.zeros((n, 4, 4), dtype=np.float64)
        L[:, 0, 0], L[:, 0, 1], L[:, 0, 3] = ct, -st, a
        L[:, 1, 0], L[:, 1, 1], L[:, 1, 2], L[:, 1, 3] = ca * st, ca * ct, -sa, -sa * d
        L[:, 2, 0], L[:, 2, 1], L[:, 2, 2], L[:, 2, 3] = sa * st, sa * ct, ca, ca * d
        L[:, 3, 3] = 1.0
        T = T @ L
    return T


def states_from_poses(Ts: np.ndarray, gs: np.ndarray, is_piper: bool) -> np.ndarray:
    align = ea.PIPER_BASE_ALIGN if is_piper else ea.NERO_BASE_ALIGN
    Tc = np.einsum("ij,njk->nik", align, Ts)
    s = np.empty((len(Ts), 10), dtype=np.float64)
    s[:, 0:3] = Tc[:, :3, 3]
    s[:, 3:6] = Tc[:, :3, 0]
    s[:, 6:9] = Tc[:, :3, 1]
    s[:, 9] = gs
    return s


def piper_valid_reason(traj: dict) -> str | None:
    src = traj.get("source") or {}
    if src.get("split") == "val":
        return "val_split"
    n = int(traj.get("num_frames") or 0)
    if n < 2:
        return "short_traj"
    pr = traj.get("proprios") or {}
    ac = traj.get("actions") or {}
    for container, key in ((pr, "left_arm_joint"), (ac, "left_arm_joint"),
                           (pr, "left_gripper_pos"), (ac, "left_gripper_pos")):
        arr = container.get(key)
        if not isinstance(arr, list) or len(arr) != n:
            return f"bad_{key}"
    obs = traj.get("observations") or {}
    if not obs.get("ego") or not obs.get("wrist_left"):
        return "missing_camera"
    prompts = (traj.get("instruction") or {}).get("general") or []
    if not prompts:
        return "missing_prompt"
    for p in prompts:
        imgs = p.get("images") or []
        convs = p.get("conversations") or []
        if "observations.wrist_right" in imgs:
            if imgs != ["observations.ego", "observations.wrist_left", "observations.wrist_right"]:
                return "unexpected_piper_cam_order"
            if not convs or convs[0].get("value", "").count("\n# Right-Wrist View\n<image>") != 1:
                return "missing_right_wrist_marker"
    return None


def make_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst)


def main() -> None:
    assert not OUT_YAML.exists(), f"{OUT_YAML} already exists"
    head = "\n".join(NERO_BASE_CFG.read_text().splitlines()[:30])
    nero_dirs = re.findall(r"^\s*-\s*(/\S+)", head, flags=re.M)
    assert len(nero_dirs) == 11, f"expected 11 NERO dirs, got {len(nero_dirs)}"

    # Verify vectorized NERO FK matches ea.nero_flange_pose
    rng = np.random.default_rng(0)
    q_test = rng.uniform(-1.0, 1.0, size=(5, 7))
    T_vec = nero_fk_batch(q_test)
    T_ref = ea.nero_flange_poses(q_test)
    assert np.allclose(T_vec, T_ref, atol=1e-12), "nero_fk_batch mismatch"

    if MIX_ROOT.exists():
        shutil.rmtree(MIX_ROOT)
    MIX_ROOT.mkdir(parents=True, exist_ok=True)

    S_ee = np.zeros((H, 6), dtype=np.float64)
    SS_ee = np.zeros((H, 6), dtype=np.float64)
    N_ee = np.zeros(H, dtype=np.float64)
    S_el = np.zeros(H, dtype=np.float64)
    SS_el = np.zeros(H, dtype=np.float64)
    N_el = np.zeros(H, dtype=np.float64)
    state_chunks = []

    nero_eps = 0
    nero_frames = 0
    for nd in nero_dirs:
        nd_p = Path(nd)
        files = sorted(Path(f) for f in glob.glob(str(nd_p / "**" / "*.json"), recursive=True))
        for f in files:
            traj = json.loads(f.read_text())
            qs = np.asarray(traj["proprios"]["arm_joint"], dtype=np.float64)
            qa = np.asarray(traj["actions"]["arm_joint"], dtype=np.float64)
            gs = np.asarray(traj["proprios"]["gripper_pos"], dtype=np.float64).reshape(len(qs))
            if not (np.all(np.isfinite(qs)) and np.all(np.isfinite(qa)) and np.all(np.isfinite(gs))):
                continue
            Ts = ea._NERO_KIN and nero_fk_batch(qs)
            Ta = nero_fk_batch(qa)
            n = len(qs)
            # Check states_from_poses matches compose_ee_state on frame 0 once
            if nero_eps == 0:
                ref_s = ea.compose_ee_state(Ts[0], gs[0], is_piper=False)[0, :10]
                assert np.allclose(states_from_poses(Ts[:1], gs[:1], False)[0], ref_s, atol=1e-6)
            st = states_from_poses(Ts, gs, is_piper=False)
            for r in range(1, NERO_REPEAT + 1):
                rel = f.relative_to(nd_p)
                make_symlink(f, MIX_ROOT / f"nero_r{r}" / nd_p.name / rel)
                state_chunks.append(st)
            for k in range(min(H, n)):
                m = n - k
                R0t = np.transpose(Ts[:m, :3, :3], (0, 2, 1))
                pos = np.einsum("nij,nj->ni", R0t, Ta[k:, :3, 3] - Ts[:m, :3, 3])
                aa = PK.rot_log_batch(R0t @ Ta[k:, :3, :3])
                v = np.concatenate([pos, aa], axis=1)
                S_ee[k] += NERO_REPEAT * v.sum(0)
                SS_ee[k] += NERO_REPEAT * (v * v).sum(0)
                N_ee[k] += NERO_REPEAT * m
                el = qa[k:, 2] - qs[:m, 2]
                S_el[k] += NERO_REPEAT * el.sum()
                SS_el[k] += NERO_REPEAT * (el * el).sum()
                N_el[k] += NERO_REPEAT * m
            nero_eps += 1
            nero_frames += n

    piper_raw = sorted(Path(f) for f in glob.glob(str(PIPER_DIR / "**" / "*.json"), recursive=True))
    skip_reasons: Counter[str] = Counter()
    piper_eps = 0
    piper_frames = 0
    for f in piper_raw:
        traj = json.loads(f.read_text())
        reason = piper_valid_reason(traj)
        if reason is not None:
            skip_reasons[reason] += 1
            continue
        qs = np.asarray(traj["proprios"]["left_arm_joint"], dtype=np.float64)
        qa = np.asarray(traj["actions"]["left_arm_joint"], dtype=np.float64)
        gs = np.asarray(traj["proprios"]["left_gripper_pos"], dtype=np.float64).reshape(len(qs)) * ea.PIPER_GRIPPER_TO_NERO_M
        if not (np.all(np.isfinite(qs)) and np.all(np.isfinite(qa)) and np.all(np.isfinite(gs))):
            skip_reasons["non_finite"] += 1
            continue
        Ts = ea.piper_flange_poses(qs)
        Ta = ea.piper_flange_poses(qa)
        n = len(qs)
        if piper_eps == 0:
            ref_s = ea.compose_ee_state(Ts[0], gs[0], is_piper=True)[0, :10]
            assert np.allclose(states_from_poses(Ts[:1], gs[:1], True)[0], ref_s, atol=1e-6)
        st = states_from_poses(Ts, gs, is_piper=True)
        state_chunks.append(st)
        rel = f.relative_to(PIPER_DIR)
        make_symlink(f, MIX_ROOT / "piper_v2" / rel)
        for k in range(min(H, n)):
            m = n - k
            R0t = np.transpose(Ts[:m, :3, :3], (0, 2, 1))
            pos = np.einsum("nij,nj->ni", R0t, Ta[k:, :3, 3] - Ts[:m, :3, 3])
            aa = PK.rot_log_batch(R0t @ Ta[k:, :3, :3])
            v = np.concatenate([pos, aa], axis=1)
            S_ee[k] += v.sum(0)
            SS_ee[k] += (v * v).sum(0)
            N_ee[k] += m
        piper_eps += 1
        piper_frames += n

    mu_ee = S_ee / N_ee[:, None]
    sd_ee = np.sqrt(np.maximum(SS_ee / N_ee[:, None] - mu_ee ** 2, 0.0))
    sd_ee = np.where(sd_ee > 1e-6, sd_ee, 1.0)

    mu_el = S_el / N_el
    sd_el = np.sqrt(np.maximum(SS_el / N_el - mu_el ** 2, 0.0))
    sd_el = np.where(sd_el > 1e-6, sd_el, 1.0)

    mean = np.zeros((H, DIM), dtype=np.float32)
    std = np.ones((H, DIM), dtype=np.float32)
    mean[:, 0:6], std[:, 0:6] = mu_ee, sd_ee
    mean[:, 6], std[:, 6] = 0.05, 0.05
    mean[:, 7], std[:, 7] = mu_el, sd_el
    mean[:, 8], std[:, 8] = 0.5, 0.5

    all_states = np.concatenate(state_chunks, axis=0)
    q01_10, q99_10 = np.percentile(all_states, [1.0, 99.0], axis=0)
    assert np.all(q99_10 > q01_10), f"invalid state quantiles: {q01_10} vs {q99_10}"
    q01 = np.zeros((1, DIM), dtype=np.float32)
    q99 = np.zeros((1, DIM), dtype=np.float32)
    q01[0, :10] = q01_10.astype(np.float32)
    q99[0, :10] = q99_10.astype(np.float32)

    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)
    assert np.all(np.isfinite(q01)) and np.all(np.isfinite(q99))

    def fmt_matrix(arr: np.ndarray, indent: int = 8) -> list[str]:
        pad = " " * indent
        lines = []
        for row in arr.tolist():
            lines.append(f"{pad}- [" + ", ".join(f"{x:.8g}" for x in row) + "]")
        return lines

    paths = [str(MIX_ROOT / f"nero_r{r}") for r in range(1, NERO_REPEAT + 1)] + [str(MIX_ROOT / "piper_v2")]
    yaml_lines = [
        "# @package _global_",
        f"# nero_ee_sb: {NERO_REPEAT}x NERO ({nero_eps} eps, {nero_frames} frames) + 1x Piper V2 ({piper_eps} eps, {piper_frames} frames)",
        "data:",
        "  type: BaseDataModule",
        "  params:",
        "    type: json",
        "    max_steps: ${trainer.max_steps}",
        "    train_datasets:",
        "      batch_size: 10",
        "      action_length: 30",
        "      action_layout: nero_ee_sb",
        "      paths:",
        *[f"      - {p}" for p in paths],
        "      mean:",
        *fmt_matrix(mean),
        "      std:",
        *fmt_matrix(std),
        "      q01:",
        *fmt_matrix(q01),
        "      q99:",
        *fmt_matrix(q99),
        "      gripper_absolute: true",
        "      dwell_keep:",
        "        default: 10",
        "        '3': 20",
        "      dwell_tail: 10",
        "      dwell_speed_deg: 2.0",
        "      dwell_gripper_m: 0.001",
    ]
    OUT_YAML.write_text("\n".join(yaml_lines) + "\n")

    indexed_counts = {Path(p).name: len(glob.glob(os.path.join(p, "**", "*.json"), recursive=True)) for p in paths}
    print(f"NERO  : eps={nero_eps}, raw_frames={nero_frames}, repeat={NERO_REPEAT}")
    print(f"Piper : eps={piper_eps}/{len(piper_raw)}, raw_frames={piper_frames}, skipped={dict(skip_reasons)}")
    print("indexed glob counts:", indexed_counts)
    print("std[0, :9] :", np.round(std[0, :9], 5).tolist())
    print("std[29, :9]:", np.round(std[29, :9], 5).tolist())
    print("q01[0, :10]:", np.round(q01[0, :10], 4).tolist())
    print("q99[0, :10]:", np.round(q99[0, :10], 4).tolist())
    print("wrote", OUT_YAML)


if __name__ == "__main__":
    main()