from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from .ee_to_nero import build_nero_ee_state, decode_nero_ee_sb_chunk


def pack_request_state_for_model(state8: np.ndarray, *, kinematics=None) -> np.ndarray:
    """Existing NERO client request state [8] -> XR1 model state [1,60]."""
    return build_nero_ee_state(state8, kinematics=kinematics)


def decode_model_action_for_client(
    de_normalized_action: np.ndarray,
    request_state8: np.ndarray,
    *,
    data_config: Mapping[str, Any],
    kinematics=None,
    **safety,
):
    """Return the exact action/StepFlag side channels expected by the NERO client."""
    result = decode_nero_ee_sb_chunk(
        de_normalized_action,
        request_state8,
        gripper_absolute=bool(data_config.get("gripper_absolute", False)),
        kinematics=kinematics,
        **safety,
    )
    return {
        "actions": result.joint_targets,
        "step_flag": result.step_flag_values,
        "step_boundary": result.step_boundary,
        "ik": result.ik_report,
    }


def build_nero_ee_sb_metadata(
    *,
    action_horizon: int,
    step_prompt_enabled: bool = False,
    gripper_absolute: bool = True,
    step_flag_threshold: float = 0.5,
    step_flag_consecutive_frames: int = 2,
) -> dict[str, Any]:
    """OpenPI metadata contract: model action dim differs from robot reply dim."""
    return {
        "model": "Xiaomi-Robotics-1",
        "protocol": "openpi-websocket-v1",
        "action_horizon": int(action_horizon),
        "action_dim": 8,                  # client receives joint7 + gripper
        "model_action_dim": 9,             # model predicts EE(8) + Step Flag(1)
        "action_layout": "nero_ee_sb",
        "camera_keys": ["front", "wrist"],
        "action_semantics": "absolute_joint",
        "state_layout": "joint7,gripper", # incoming client request
        "model_state_layout": "ee_flange_pos3_rot6d6_gripper1",
        "model_state_valid_dim": 10,
        "gripper_range": [0.0, 0.1],
        "gripper_mode": "absolute" if gripper_absolute else "delta",
        "rtc": {"mode": "disabled", "max_prefix_steps": 0},
        "step_prompt": {"enabled": bool(step_prompt_enabled)},
        "step_flag": {
            "enabled": True,
            "channel": 8,
            "threshold": float(step_flag_threshold),
            "consecutive_frames": int(step_flag_consecutive_frames),
            "hold_after_boundary": True,
            "manual_step_confirmation": True,
        },
    }