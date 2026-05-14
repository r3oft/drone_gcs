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
        return self.process_frame_all(frame, [target_cls_id]).get(target_cls_id)

    def process_frame_all(self, frame, target_cls_ids=None):
        self.calls += 1
        targets = self._targets()
        if target_cls_ids is None:
            return targets
        return {cls_id: targets[cls_id] for cls_id in target_cls_ids if cls_id in targets}

    def _targets(self):
        return {
            0: {
                "cls_id": 0,
                "u": 160.0 + (self.calls % 5),
                "v": 120.0,
                "theta": 0.05,
                "conf": 0.92,
                "w": 64.0,
                "h": 40.0,
            },
            1: {
                "cls_id": 1,
                "u": 236.0,
                "v": 150.0,
                "theta": -0.12,
                "conf": 0.84,
                "w": 54.0,
                "h": 48.0,
            },
        }


class PickupOnlyEstimator(FakeEstimator):
    def _targets(self):
        return {0: super()._targets()[0]}


class SpyVisualizer(DebugVisualizer):
    last_instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(record_path=None, snapshot_dir=None)
        self.frames = []
        self.hud_infos = []
        self.obb_labels = []
        self.obb_thicknesses = []
        self.error_targets = []
        SpyVisualizer.last_instance = self

    def draw_obb(self, frame, *args, **kwargs):
        self.obb_labels.append(kwargs.get("label", ""))
        self.obb_thicknesses.append(kwargs.get("thickness", 2))
        return super().draw_obb(frame, *args, **kwargs)

    def draw_error_vector(self, frame, center, target, *args, **kwargs):
        self.error_targets.append(target)
        return super().draw_error_vector(frame, center, target, *args, **kwargs)

    def draw_hud(self, frame, info):
        self.hud_infos.append(dict(info))
        return super().draw_hud(frame, info)

    def write_frame(self, frame):
        self.frames.append(frame.copy())
        super().write_frame(frame)


def run_hud_once(monkeypatch, tmp_path, target="pickup_zone", estimator_cls=FakeEstimator):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"fake weights")

    FakeStreamer.last_instance = None
    SpyVisualizer.last_instance = None

    fake_perception = types.SimpleNamespace(TargetPoseEstimator=estimator_cls)
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
            "--target",
            target,
            "--draw-targets",
            "all",
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
    assert visualizer is not None
    return streamer, visualizer


def test_split_record_paths_places_outputs_under_hud_subdirs():
    with_hud, non_hud = run_visual_hud.split_record_paths(
        "logs/hud_live_test/live_hud.avi"
    )

    assert with_hud == "logs/hud_live_test/with_hud/live_hud.avi"
    assert non_hud == "logs/hud_live_test/non_hud/live_hud.avi"


def test_pickup_target_controls_hud_velocity_and_error_arrow(monkeypatch, tmp_path):
    streamer, visualizer = run_hud_once(monkeypatch, tmp_path, target="pickup_zone")

    assert streamer.released is True
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

    assert "pickup_zone CTRL" in visualizer.obb_labels
    assert "delivery_zone" in visualizer.obb_labels
    assert 3 in visualizer.obb_thicknesses
    assert visualizer.error_targets
    assert visualizer.error_targets[0][1] == 120
    assert 160 <= visualizer.error_targets[0][0] <= 164

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


def test_delivery_target_controls_hud_velocity_and_error_arrow(monkeypatch, tmp_path):
    _, visualizer = run_hud_once(monkeypatch, tmp_path, target="delivery_zone")

    first_hud = visualizer.hud_infos[0]
    assert first_hud["target"] == "delivery_zone"
    assert first_hud["conf"] == 0.84
    assert first_hud["vy"] > 0.0
    assert first_hud["vyaw"] > 0.0
    assert first_hud["err_y"] == 76.0
    assert first_hud["err_yaw"] == 0.12

    assert "pickup_zone" in visualizer.obb_labels
    assert "delivery_zone CTRL" in visualizer.obb_labels
    assert visualizer.error_targets
    assert visualizer.error_targets[0] == (236, 150)


def test_missing_active_target_keeps_velocity_zero_without_arrow(monkeypatch, tmp_path):
    _, visualizer = run_hud_once(
        monkeypatch,
        tmp_path,
        target="delivery_zone",
        estimator_cls=PickupOnlyEstimator,
    )

    first_hud = visualizer.hud_infos[0]
    assert first_hud["target"] == "none"
    assert first_hud["vx"] == 0.0
    assert first_hud["vy"] == 0.0
    assert first_hud["vyaw"] == 0.0
    assert "conf" not in first_hud
    assert "err_x" not in first_hud

    assert "pickup_zone" in visualizer.obb_labels
    assert "delivery_zone CTRL" not in visualizer.obb_labels
    assert visualizer.error_targets == []


def test_record_path_writes_with_hud_and_non_hud_videos(monkeypatch, tmp_path):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"fake weights")
    record_path = tmp_path / "live_hud.avi"

    fake_perception = types.SimpleNamespace(TargetPoseEstimator=FakeEstimator)
    monkeypatch.setitem(sys.modules, "core.perception", fake_perception)
    monkeypatch.setattr(run_visual_hud, "ZeroLatencyStreamer", FakeStreamer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_visual_hud.py",
            "--weights",
            str(weights),
            "--device",
            "cpu",
            "--record-path",
            str(record_path),
            "--duration",
            "0.25",
            "--infer-hz",
            "20",
            "--output-hz",
            "20",
            "--report-interval-s",
            "10.0",
            "--detection-stale-s",
            "1.0",
        ],
    )

    run_visual_hud.main()

    with_hud_path = tmp_path / "with_hud" / "live_hud.avi"
    non_hud_path = tmp_path / "non_hud" / "live_hud.avi"
    assert with_hud_path.is_file()
    assert non_hud_path.is_file()
    assert with_hud_path.stat().st_size > 0
    assert non_hud_path.stat().st_size > 0

    with_hud_cap = cv2.VideoCapture(str(with_hud_path))
    non_hud_cap = cv2.VideoCapture(str(non_hud_path))
    with_hud_ok, with_hud_frame = with_hud_cap.read()
    non_hud_ok, non_hud_frame = non_hud_cap.read()
    with_hud_cap.release()
    non_hud_cap.release()

    assert with_hud_ok
    assert non_hud_ok
    assert np.max(with_hud_frame[:140, :270]) > 180
    assert np.max(non_hud_frame[:140, :270]) < 120
