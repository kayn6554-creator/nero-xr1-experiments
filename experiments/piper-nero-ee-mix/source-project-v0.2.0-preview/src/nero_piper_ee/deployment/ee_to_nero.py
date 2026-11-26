from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import time

import numpy as np


@dataclass(frozen=True)
class NeroEEChunk:
    """Decoded result for the existing NERO client.

    `joint_targets` is [T,8]: seven absolute joint targets in radians plus
    absolute gripper target in metres. Step Flag remains a side channel.
    """

    joint_targets: np.ndarray
    step_flag_values: np.ndarray
    step_transition: np.ndarray
    step_boundary: dict[str, Any]
    ik_attempted: np.ndarray
    ik_success: np.ndarray
    ik_position_error_m: np.ndarray
    ik_rotation_error_rad: np.ndarray
    ik_report: dict[str, Any]


def _resolve_kinematics(kinematics=None):
    if kinematics is not None:
        return kinematics
    try:
        from mibot.utils import nero_kin
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import mibot.utils.nero_kin. Put the XR1/NERO overlay on PYTHONPATH "
            "or pass a kinematics module explicitly."
        ) from exc
    return nero_kin


def build_nero_ee_state(
    current_nero_state: np.ndarray,
    *,
    kinematics=None,
    state_composer=None,
) -> np.ndarray:
    """Convert client state [joint1..joint7, gripper_m] to training State [1,60].

    This deliberately calls the same `compose_ee_state()` used by
    `JsonDataset._state()` for NERO trajectories. It therefore preserves the
    training rotation-6D convention and NERO_BASE_ALIGN setting.
    """
    state = np.asarray(current_nero_state, dtype=np.float64)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("current_nero_state must be finite shape [8]: joint7 + gripper_m")

    kin = _resolve_kinematics(kinematics)
    T0 = np.asarray(kin.fk(state[:7]), dtype=np.float64)
    if T0.shape != (4, 4) or not np.isfinite(T0).all():
        raise RuntimeError("NERO FK returned an invalid 4x4 flange pose")

    if state_composer is None:
        try:
            from mibot.utils.ef_align import compose_ee_state
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import the training-side compose_ee_state() from mibot.utils.ef_align"
            ) from exc
        state_composer = compose_ee_state

    packed = np.asarray(state_composer(T0, float(state[7]), is_piper=False), dtype=np.float32)
    if packed.shape != (1, 60) or not np.isfinite(packed).all():
        raise RuntimeError(f"compose_ee_state returned invalid shape/value: {packed.shape}")
    return packed


def detect_step_flag_boundary(
    step_flag_values: np.ndarray,
    *,
    threshold: float = 0.5,
    consecutive_frames: int = 2,
) -> dict[str, Any]:
    """Find the first run of `consecutive_frames` de-normalized Step Flag values.

    Boundary index k means [0,k) is the current step; k and later are held.
    `trigger_index` is the last frame in the consecutive run that confirmed it.
    """
    flags = np.asarray(step_flag_values, dtype=np.float64)
    if flags.ndim != 1 or len(flags) < 1 or not np.isfinite(flags).all():
        raise ValueError("step_flag_values must be a non-empty finite 1-D array")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if int(consecutive_frames) < 1:
        raise ValueError("consecutive_frames must be >= 1")

    run = int(consecutive_frames)
    boundary_index = None
    trigger_index = None
    for start in range(max(0, len(flags) - run + 1)):
        if np.all(flags[start:start + run] >= threshold):
            boundary_index = start
            trigger_index = start + run - 1
            break

    return {
        "enabled": True,
        "detected": boundary_index is not None,
        "boundary_index": boundary_index,
        "trigger_index": trigger_index,
        "threshold": float(threshold),
        "consecutive_frames": run,
        "max_value": float(np.max(flags)),
        "first_value": float(flags[0]),
        "hold_after_boundary": boundary_index is not None,
        "manual_step_confirmation": boundary_index is not None,
    }


def apply_step_flag_boundary(
    joint_actions: np.ndarray,
    current_nero_state: np.ndarray,
    step_flag_values: np.ndarray,
    *,
    threshold: float = 0.5,
    consecutive_frames: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the existing M5 hold rule to [T,8] absolute joint targets."""
    actions = np.asarray(joint_actions, dtype=np.float64)
    state = np.asarray(current_nero_state, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
        raise ValueError("joint_actions must be finite shape [T,8]")
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("current_nero_state must be finite shape [8]")
    if len(actions) != len(step_flag_values):
        raise ValueError("joint_actions and step_flag_values must have the same horizon")

    info = detect_step_flag_boundary(
        step_flag_values, threshold=threshold, consecutive_frames=consecutive_frames
    )
    out = actions.copy()
    k = info["boundary_index"]
    if k is not None:
        if k == 0:
            out[:] = state[None, :]
        else:
            out[k:] = out[k - 1]
    return out.astype(np.float32), info


def decode_nero_ee_sb_chunk(
    action_chunk: np.ndarray,
    current_nero_state: np.ndarray,
    *,
    gripper_absolute: bool = True,
    step_flag_threshold: float = 0.5,
    step_flag_consecutive_frames: int = 2,
    max_joint_jump_deg: float | None = 15.0,
    tol_pos_m: float = 0.0001,
    tol_rot_rad: float = 0.001,
    ik_kwargs: Mapping[str, Any] | None = None,
    kinematics=None,
) -> NeroEEChunk:
    """Decode de-normalized `nero_ee_sb` [T,60] to NERO absolute joint targets.

    Action slots:
      0:3 local flange-frame translation delta (m)
      3:6 local flange-frame rotation vector delta (rad)
      6   gripper target (m if absolute; otherwise delta m)
      7   NERO j3/elbow delta (rad, relative to chunk-start j3)
      8   de-normalized Step Flag (0/1 target)

    Step Flag detection occurs before IK. If a boundary is found at k, only
    [0,k) is solved; k and later hold the last valid target (or current state
    when k=0). This avoids rejecting a chunk because of unreachable actions
    that the client must not execute after the boundary.
    """
    actions = np.asarray(action_chunk, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 60 or actions.shape[0] < 1:
        raise ValueError(f"action_chunk must be [T,60], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("action_chunk contains NaN or Inf")

    state = np.asarray(current_nero_state, dtype=np.float64)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("current_nero_state must be finite shape [8]")

    flags = actions[:, 8].copy()
    boundary = detect_step_flag_boundary(
        flags,
        threshold=step_flag_threshold,
        consecutive_frames=step_flag_consecutive_frames,
    )
    valid_steps = boundary["boundary_index"] if boundary["detected"] else len(actions)

    kin = _resolve_kinematics(kinematics)
    q0 = state[:7].copy()
    g0 = float(state[7])
    T0 = np.asarray(kin.fk(q0), dtype=np.float64)
    if T0.shape != (4, 4) or not np.isfinite(T0).all():
        raise RuntimeError("NERO FK returned an invalid flange pose")

    kwargs = {
        "iters": 100,
        "tol_pos": float(tol_pos_m),
        "tol_rot": float(tol_rot_rad),
        "tol_q3": float(tol_rot_rad),
        "damping": 0.02,
        "null_gain": 0.2,
        "max_step": 0.2,
    }
    if ik_kwargs:
        forbidden = {"q_ref", "q3_target"} & set(ik_kwargs)
        if forbidden:
            raise ValueError(f"ik_kwargs cannot override managed fields: {sorted(forbidden)}")
        kwargs.update(dict(ik_kwargs))

    t_start = time.perf_counter()
    out = np.repeat(state[None, :], len(actions), axis=0)
    attempted = np.zeros(len(actions), dtype=bool)
    success = np.ones(len(actions), dtype=bool)  # held frames are valid outputs, not failed IK attempts
    pos_errors = np.full(len(actions), np.nan, dtype=np.float64)
    rot_errors = np.full(len(actions), np.nan, dtype=np.float64)
    q_seed = q0.copy()
    max_jump = 0.0
    max_pos = 0.0
    max_rot = 0.0
    jump_at = ""

    for t in range(valid_steps):
        row = actions[t]
        T_delta = np.eye(4, dtype=np.float64)
        T_delta[:3, :3] = kin.rot_exp(row[3:6])
        T_delta[:3, 3] = row[:3]
        # Training stores T_start^-1 @ T_future, so recover T_future = T_start @ T_delta.
        T_target = T0 @ T_delta
        # Slot 7 is q[j3] - q_start[j3], not an absolute j3 value.
        q3_target = float(q0[2] + row[7])

        q_target, ok, ep, er = kin.ik(
            T_target,
            q_seed,
            q3_target=q3_target,
            **kwargs,
        )
        q_target = np.asarray(q_target, dtype=np.float64)
        attempted[t] = True
        success[t] = bool(ok) and q_target.shape == (7,) and np.isfinite(q_target).all()
        pos_errors[t] = float(ep)
        rot_errors[t] = float(er)

        if not success[t]:
            raise RuntimeError(
                f"NERO EE-SB IK failed at horizon index {t}: "
                f"position_error={float(ep) * 1000:.3f} mm, "
                f"rotation_error={np.degrees(float(er)):.4f} deg"
            )

        jump_vec = np.abs(q_target - q_seed)
        jump = float(np.degrees(jump_vec.max()))
        if jump >= max_jump:
            jump_at = f"k{t}j{int(np.argmax(jump_vec)) + 1}"
        max_jump = max(max_jump, jump)
        if max_joint_jump_deg is not None and jump > float(max_joint_jump_deg):
            raise RuntimeError(
                f"NERO EE-SB IK joint jump {jump:.2f} deg at horizon index {t} "
                f"> {float(max_joint_jump_deg):.2f} deg"
            )

        max_pos = max(max_pos, float(ep) * 1000.0)
        max_rot = max(max_rot, float(np.degrees(er)))
        gripper_target = float(row[6]) if gripper_absolute else g0 + float(row[6])
        out[t] = np.r_[q_target, gripper_target]
        q_seed = q_target

    if boundary["detected"]:
        k = int(boundary["boundary_index"])
        if k == 0:
            out[:] = state[None, :]
        else:
            out[k:] = out[k - 1]

    ik_report = {
        "max_pos_err_mm": round(max_pos, 4),
        "max_rot_err_deg": round(max_rot, 4),
        "max_step_jump_deg": round(max_jump, 4),
        "ms": round((time.perf_counter() - t_start) * 1000.0, 1),
        "jump_at": jump_at,
        "solved_steps": int(attempted.sum()),
        "held_steps": int(len(actions) - valid_steps) if boundary["detected"] else 0,
        "boundary_index": boundary["boundary_index"],
    }

    return NeroEEChunk(
        joint_targets=out.astype(np.float32),
        step_flag_values=flags.astype(np.float32),
        step_transition=(flags >= float(step_flag_threshold)),
        step_boundary=boundary,
        ik_attempted=attempted,
        ik_success=success,
        ik_position_error_m=pos_errors.astype(np.float32),
        ik_rotation_error_rad=rot_errors.astype(np.float32),
        ik_report=ik_report,
    )
