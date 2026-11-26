from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "collector"))
import nero_depth


class FakeWriter:
    instances = []

    def __init__(self, path, fps, w, h):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        self.n = 0
        self.err = None
        self.s = SimpleNamespace(width=w, height=h)
        self.frames = []
        self.closed = False
        self.__class__.instances.append(self)

    def put(self, frame):
        arr = np.ascontiguousarray(frame, dtype=np.uint16)
        self.frames.append(arr.copy())
        self.n += 1

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_fake_writer(monkeypatch):
    FakeWriter.instances = []
    monkeypatch.setattr(nero_depth, "_Writer", FakeWriter)


def test_leading_and_interior_missing_depth_are_zero_filled(tmp_path):
    recorder = nero_depth.DepthRecorder(tmp_path, ["front", "wrist"], fps=15)
    recorder.start()

    front_1 = np.full((2, 3), 1200, dtype=np.uint16)
    wrist_2 = np.full((2, 3), 2400, dtype=np.uint16)
    recorder.add({"front": None, "wrist": None})
    recorder.add({"front": front_1, "wrist": None})
    recorder.add({"front": None, "wrist": wrist_2})

    result = recorder.finish(episode=7, n_frames=3)

    assert result["front"] == {"frames": 3, "ok": True, "error": None}
    assert result["wrist"] == {"frames": 3, "ok": True, "error": None}
    front, wrist = FakeWriter.instances
    np.testing.assert_array_equal(front.frames[0], np.zeros((2, 3), np.uint16))
    np.testing.assert_array_equal(front.frames[1], front_1)
    np.testing.assert_array_equal(front.frames[2], np.zeros((2, 3), np.uint16))
    np.testing.assert_array_equal(wrist.frames[0], np.zeros((2, 3), np.uint16))
    np.testing.assert_array_equal(wrist.frames[1], np.zeros((2, 3), np.uint16))
    np.testing.assert_array_equal(wrist.frames[2], wrist_2)
    assert (tmp_path / "depth/front/episode_000007.mkv").is_file()
    assert (tmp_path / "depth/wrist/episode_000007.mkv").is_file()


def test_camera_with_no_valid_depth_is_reported_as_missing(tmp_path):
    recorder = nero_depth.DepthRecorder(tmp_path, ["front"], fps=15)
    recorder.start()
    recorder.add({"front": None})
    recorder.add({"front": None})

    result = recorder.finish(episode=8, n_frames=2)

    assert result["front"]["frames"] == 0
    assert result["front"]["ok"] is False
    assert "no valid depth" in result["front"]["error"]
    assert not (tmp_path / "depth/front/episode_000008.mkv").exists()


def test_finish_detects_rgb_depth_frame_count_mismatch(tmp_path):
    recorder = nero_depth.DepthRecorder(tmp_path, ["front"], fps=15)
    recorder.start()
    recorder.add({"front": np.ones((2, 3), dtype=np.uint16)})

    result = recorder.finish(episode=9, n_frames=2)

    assert result["front"]["frames"] == 1
    assert result["front"]["ok"] is False
    assert "expected 2" in result["front"]["error"]


def test_depth_shape_change_is_rejected_before_writing_current_frame(tmp_path):
    recorder = nero_depth.DepthRecorder(tmp_path, ["front"], fps=15)
    recorder.start()
    recorder.add({"front": np.ones((2, 3), dtype=np.uint16)})

    with pytest.raises(ValueError, match="shape changed"):
        recorder.add({"front": np.ones((3, 3), dtype=np.uint16)})

    assert recorder.frame_count == 1
    assert recorder.cur["front"].n == 1
    recorder.abort()
