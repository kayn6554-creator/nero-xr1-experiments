from scripts.validation.mix_dataset_utils import count_dwell_kept


def test_dwell_filter_keeps_head_and_tail_within_step():
    n = 10
    traj = {"source": {"step_index": [0] * n}}
    q = __import__("numpy").zeros((n, 6))
    g = __import__("numpy").zeros(n)
    cfg = {
        "dwell_keep": {"default": 2},
        "dwell_tail": 2,
        "dwell_speed_deg": 2.0,
        "dwell_gripper_m": 0.001,
    }
    kept, removed = count_dwell_kept(traj, q, g, n, "piper", cfg)
    # Frame 0 is non-static due the leading +inf; frames 1..9 form one
    # static run. Keep first 2 and last 2 of that run.
    assert kept == 5
    assert removed == 5