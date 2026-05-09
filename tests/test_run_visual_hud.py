import sys
import types

import cv2
import numpy as np

import scripts.run_visual_hud as run_visual_hud
from utils.visualization import DebugVisualizer


class FakeStreamer:
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.released = False
        FakeStreamer.last_instance = self

    @property
    def current_mode(self):
        return "stream"

    @property
    def frame_update_count(self):
        return self.calls

    def get_latest_frame(self):
        self.calls += 1
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :] = (self.calls % 255, 24, 48)
        cv2.putText(
            frame,
            f"src {self.calls}",
            (220, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        return frame

    def release(self):
        self.released = True


class FakeEstimator:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def process_frame(self, frame, target_cls_id):
        self.calls += 1
        return {
            "u": 160.0 + (self.calls % 5),
            "v": 120.0,
            "theta": 0.05,
            "conf": 0.92,
            "w": 64.0,
            "h": 40.0,
        }


class SpyVisualizer(DebugVisualizer):
    last_instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(record_path=None, snapshot_dir=None)
        self.frames = []
        self.hud_infos = []
        SpyVisualizer.last_instance = self

    def draw_hud(self, frame, info):
        self.hud_infos.append(dict(info))
        return super().draw_hud(frame, info)

    def write_frame(self, frame):
        self.frames.append(frame.copy())
        super().write_frame(frame)


def test_run_visual_hud_outputs_continuous_hud_annotated_frames(
    monkeypatch, tmp_path
):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"fake weights")

    fake_perception = types.SimpleNamespace(TargetPoseEstimator=FakeEstimator)
    monkeypatch.setitem(sys.modules, "core.perception", fake_perception)
    monkeypatch.setattr(run_visual_hud, "ZeroLatencyStreamer", FakeStreamer)
    monkeypatch.setattr(run_visual_hud, "DebugVisualizer", SpyVisualizer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_visual_hud.py",
            "--weights",
            str(weights),
            "--device",
            "cpu",
            "--duration",
            "0.35",
            "--infer-hz",
            "20",
            "--output-hz",
            "20",
            "--report-interval-s",
            "0.05",
            "--detection-stale-s",
            "1.0",
        ],
    )

    run_visual_hud.main()

    streamer = FakeStreamer.last_instance
    visualizer = SpyVisualizer.last_instance

    assert streamer is not None
    assert streamer.released is True
    assert visualizer is not None
    assert len(visualizer.frames) >= 3
    assert len(visualizer.hud_infos) == len(visualizer.frames)

    first_hud = visualizer.hud_infos[0]
    assert first_hud["state"] == "VISUAL_HUD"
    assert first_hud["mode"] == "stream"
    assert first_hud["target"] == "pickup_zone"
    assert first_hud["conf"] == 0.92
    assert "err_x" in first_hud
    assert "p_x" in first_hud
    assert "d_x" in first_hud

    # The HUD draws white text in the upper-left panel; the synthetic source is
    # deliberately dark, so bright pixels here prove HUD information was added.
    assert all(np.max(frame[:140, :270]) > 180 for frame in visualizer.frames[:3])

    # The untouched lower-right source area should change across written frames,
    # proving the runner is producing a continuous frame sequence.
    source_values = {
        int(frame[220, 300, 0])
        for frame in visualizer.frames
    }
    assert len(source_values) >= 2
