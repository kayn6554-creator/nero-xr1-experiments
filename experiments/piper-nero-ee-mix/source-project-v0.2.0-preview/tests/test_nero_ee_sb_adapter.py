import numpy as np
import pytest

from mibot.utils import nero_kin
from mibot.utils.ef_align import compose_ee_state
from nero_piper_ee.deployment.ee_to_nero import (
    apply_step_flag_boundary,
    build_nero_ee_state,
    decode_nero_ee_sb_chunk,
    detect_step_flag_boundary,
)
from nero_piper_ee.deployment.nero_ee_sb_server_adapter import (
    build_nero_ee_sb_metadata,
    pack_request_state_for_model,
)


def _sample_chunk(q0, q1, flag=None):
    T0, T1 = nero_kin.fk(q0), nero_kin.fk(q1)
    pos = T0[:3, :3].T @ (T1[:3, 3] - T0[:3, 3])
    aa = nero_kin.rot_log(T0[:3, :3].T @ T1[:3, :3])
    chunk = np.zeros((30, 60), dtype=np.float64)
    chunk[:, 0:3] = pos
    chunk[:, 3:6] = aa
    chunk[:, 6] = 0.045
    chunk[:, 7] = q1[2] - q0[2]
    if flag is not None:
        chunk[:, 8] = flag
    return chunk


def test_request_state_matches_training_compose_ee_state():
    q = np.array([0.1, -0.4, 0.5, 0.2, 0.3, 0.1, 0.0])
    state8 = np.r_[q, 0.035]
    T = nero_kin.fk(q)
    expected = compose_ee_state(T, 0.035, is_piper=False)
    actual = build_nero_ee_state(state8)
    assert actual.shape == (1, 60)
    assert np.allclose(actual, expected, atol=1e-7)
    assert np.allclose(pack_request_state_for_model(state8), expected, atol=1e-7)


def test_step_flag_requires_two_consecutive_frames():
    one_spike = np.zeros(30)
    one_spike[5] = 1.0
    assert not detect_step_flag_boundary(one_spike)["detected"]

    flag = np.zeros(30)
    flag[4:6] = 1.0
    info = detect_step_flag_boundary(flag)
    assert info["detected"]
    assert info["boundary_index"] == 4
    assert info["trigger_index"] == 5


def test_step_flag_boundary_holds_last_valid_joint_target():
    current = np.arange(8, dtype=np.float64)
    predicted = np.arange(30 * 8, dtype=np.float64).reshape(30, 8)
    flag = np.zeros(30)
    flag[3:5] = 1.0
    held, info = apply_step_flag_boundary(predicted, current, flag)
    assert info["boundary_index"] == 3
    assert np.array_equal(held[3:], np.repeat(held[2:3], 27, axis=0))

    flag[:] = 1.0
    held0, info0 = apply_step_flag_boundary(predicted, current, flag)
    assert info0["boundary_index"] == 0
    assert np.array_equal(held0, np.repeat(current[None], 30, axis=0))


def test_nero_ee_sb_decoder_solves_prefix_then_holds():
    q0 = np.array([0.1, -0.4, 0.5, 0.2, 0.3, 0.1, 0.0], dtype=np.float64)
    q1 = q0 + np.array([0.02, -0.015, 0.01, 0.02, -0.01, 0.015, 0.005])
    chunk = _sample_chunk(q0, q1)
    chunk[4:6, 8] = 1.0

    result = decode_nero_ee_sb_chunk(chunk, np.r_[q0, 0.02], gripper_absolute=True)
    assert result.step_boundary["detected"]
    assert result.step_boundary["boundary_index"] == 4
    assert result.ik_attempted[:4].all()
    assert not result.ik_attempted[4:].any()
    assert np.allclose(result.joint_targets[4:], result.joint_targets[3:4])
    assert result.ik_report["solved_steps"] == 4
    assert result.ik_report["held_steps"] == 26


def test_boundary_at_zero_holds_current_state_without_ik():
    q0 = np.array([0.1, -0.4, 0.5, 0.2, 0.3, 0.1, 0.0], dtype=np.float64)
    chunk = _sample_chunk(q0, q0)
    chunk[:, 8] = 1.0
    state8 = np.r_[q0, 0.025]
    result = decode_nero_ee_sb_chunk(chunk, state8)
    assert result.step_boundary["boundary_index"] == 0
    assert not result.ik_attempted.any()
    assert np.allclose(result.joint_targets, np.repeat(state8[None], 30, axis=0))


def test_metadata_separates_step_flag_from_step_prompt():
    metadata = build_nero_ee_sb_metadata(action_horizon=30, step_prompt_enabled=False)
    assert metadata["action_dim"] == 8
    assert metadata["model_action_dim"] == 9
    assert metadata["step_flag"]["enabled"] is True
    assert metadata["step_prompt"]["enabled"] is False
    assert metadata["state_layout"] == "joint7,gripper"
    assert metadata["model_state_layout"] == "ee_flange_pos3_rot6d6_gripper1"
