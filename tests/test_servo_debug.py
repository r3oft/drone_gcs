import pytest

from core.servo_controller import VisualServoController


DEFAULT_KP = [0.003, 0.003, 0.6]
DEFAULT_MV = [0.3, 0.3, 0.5]
CU, CV = 320.0, 240.0


def make_pose(u=CU, v=CV, theta=0.0, conf=0.9):
    return {"u": u, "v": v, "theta": theta, "conf": conf}


def test_compute_debug_exposes_pd_terms_for_hud():
    ctrl = VisualServoController(
        kp=DEFAULT_KP,
        kd=[0.0, 0.0, 0.0],
        deadband=[0.0, 0.0, 0.0],
        max_vel=DEFAULT_MV,
    )

    debug = ctrl.compute_debug(make_pose(u=360, v=200, theta=0.1), CU, CV, dt=0.1)

    assert set(debug) >= {
        "raw_errors",
        "errors",
        "p_terms",
        "d_terms",
        "velocities_raw",
        "velocities",
    }
    assert len(debug["velocities"]) == 3
    assert debug["errors"] == pytest.approx((40.0, 40.0, -0.1))


def test_compute_velocity_still_returns_velocity_tuple():
    ctrl = VisualServoController(
        kp=DEFAULT_KP,
        kd=[0.0, 0.0, 0.0],
        deadband=[0.0, 0.0, 0.0],
        max_vel=DEFAULT_MV,
    )

    velocity = ctrl.compute_velocity(make_pose(v=200), CU, CV, dt=0.1)

    assert isinstance(velocity, tuple)
    assert len(velocity) == 3
