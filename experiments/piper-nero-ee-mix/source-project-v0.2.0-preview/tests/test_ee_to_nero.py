import numpy as np
import pytest

from nero_piper_ee.deployment.ee_to_nero import decode_nero_ee_sb_chunk
from mibot.utils import nero_kin


def test_ee_action_round_trip_to_nero_joint_targets():
    q0 = np.array([0.1, -0.4, 0.5, 0.2, 0.3, 0.1, 0.0], dtype=np.float64)
    q1 = q0 + np.array([0.02, -0.015, 0.01, 0.02, -0.01, 0.015, 0.005])
    T0, T1 = nero_kin.fk(q0), nero_kin.fk(q1)

    local_pos = T0[:3, :3].T @ (T1[:3, 3] - T0[:3, 3])
    local_rot = nero_kin.rot_log(T0[:3, :3].T @ T1[:3, :3])

    chunk = np.zeros((30, 60), dtype=np.float64)
    chunk[:, 0:3] = local_pos
    chunk[:, 3:6] = local_rot
    chunk[:, 6] = 0.04
    chunk[:, 7] = q1[2] - q0[2]
    chunk[:, 8] = 0.0

    result = decode_nero_ee_sb_chunk(
        chunk,
        np.r_[q0, 0.02],
        gripper_absolute=True,
        max_joint_jump_deg=15.0,
    )

    assert result.joint_targets.shape == (30, 8)
    assert result.ik_attempted.all()
    assert result.ik_success.all()
    assert not result.step_boundary["detected"]
    assert result.joint_targets[0, 7] == pytest.approx(0.04)
    assert result.joint_targets[0, 2] == pytest.approx(q1[2], abs=2e-4)
    T_recovered = nero_kin.fk(result.joint_targets[0, :7])
    assert np.linalg.norm(T_recovered[:3, 3] - T1[:3, 3]) < 1e-3
    assert np.linalg.norm(nero_kin.rot_log(T_recovered[:3, :3] @ T1[:3, :3].T)) < 1e-3


def test_rejects_wrong_action_shape():
    with pytest.raises(ValueError):
        decode_nero_ee_sb_chunk(np.zeros((30, 9)), np.zeros(8))


def test_rejects_non_finite_action():
    chunk = np.zeros((2, 60))
    chunk[0, 0] = np.nan
    with pytest.raises(ValueError):
        decode_nero_ee_sb_chunk(chunk, np.zeros(8))