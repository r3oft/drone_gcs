import numpy as np

from utils.visualization import DebugVisualizer


def test_hud_accepts_live_runner_debug_fields():
    viz = DebugVisualizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    info = {
        "state": "VISUAL_HUD",
        "camera": "esp32_s3",
        "mode": "stream",
        "target": "pickup_zone",
        "conf": 0.91,
        "vx": 0.12,
        "vy": -0.04,
        "vyaw": 0.03,
        "err_x": 22.0,
        "err_y": -18.0,
        "err_yaw": 0.08,
        "p_x": 0.066,
        "p_y": -0.054,
        "p_yaw": 0.048,
        "d_x": 0.010,
        "d_y": -0.006,
        "d_yaw": 0.001,
        "source_fps": 26.2,
        "read_fps": 860.0,
    }

    result = viz.draw_hud(frame, info)

    assert result is frame
    assert np.any(frame != 0)
