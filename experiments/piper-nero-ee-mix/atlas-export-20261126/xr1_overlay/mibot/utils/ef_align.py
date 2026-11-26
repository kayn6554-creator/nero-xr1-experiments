"""Shared Piper/NERO end-effector alignment and unified EE state helpers.

Pure NumPy module: safe to import without Torch/mibot package initialization.
Unverified hardware calibration constants are isolated at the top of this file.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import numpy as np

# [待真机/腕相机朝向验证] Fixed rotation aligning Piper link6 frame to NERO flange frame.
PIPER_TO_NERO_FLANGE_R = np.eye(3, dtype=np.float64)
# [待真机验证] Linear scale mapping Piper normalized gripper [0..1] -> NERO [0..0.1] equivalent.
PIPER_GRIPPER_TO_NERO_M = 0.1
# [待真机/工作空间验证] Optional canonical base alignment for unified EE state.
PIPER_BASE_ALIGN = np.eye(4, dtype=np.float64)
NERO_BASE_ALIGN = np.eye(4, dtype=np.float64)


def _load_sibling(name: str):
    full = f"mibot.utils.{name}"
    if full in sys.modules:
        return sys.modules[full]
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_NERO_KIN = _load_sibling("nero_kin")
_PIPER_KIN = _load_sibling("piper_kin")

_T_PIPER_FLANGE = np.eye(4, dtype=np.float64)
_T_PIPER_FLANGE[:3, :3] = PIPER_TO_NERO_FLANGE_R


def is_piper_traj(traj: dict) -> bool:
    proprios = traj.get("proprios") or {}
    return "left_arm_joint" in proprios and "arm_joint" not in proprios


def nero_flange_pose(q) -> np.ndarray:
    return _NERO_KIN.fk(np.asarray(q, dtype=np.float64))


def nero_flange_poses(qs) -> np.ndarray:
    qs = np.asarray(qs, dtype=np.float64)
    return np.stack([_NERO_KIN.fk(q) for q in qs])


def piper_flange_pose(q) -> np.ndarray:
    return _PIPER_KIN.fk(np.asarray(q, dtype=np.float64)) @ _T_PIPER_FLANGE


def piper_flange_poses(qs) -> np.ndarray:
    return _PIPER_KIN.fk_batch(np.asarray(qs, dtype=np.float64)) @ _T_PIPER_FLANGE


def ee_relative_action(T0: np.ndarray, Ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R0t = T0[:3, :3].T
    pos = (R0t @ (Ts[:, :3, 3] - T0[:3, 3]).T).T
    aa = _PIPER_KIN.rot_log_batch(np.einsum("ij,njk->nik", R0t, Ts[:, :3, :3]))
    return pos, aa


def compose_ee_state(T0: np.ndarray, gripper_m: float, is_piper: bool = False) -> np.ndarray:
    align = PIPER_BASE_ALIGN if is_piper else NERO_BASE_ALIGN
    Tc = align @ T0
    state = np.zeros((1, 60), dtype=np.float32)
    state[0, 0:3] = Tc[:3, 3]
    state[0, 3:9] = _PIPER_KIN.rot6d(Tc[:3, :3])
    state[0, 9] = float(gripper_m)
    return state
