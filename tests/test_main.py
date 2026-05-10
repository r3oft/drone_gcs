import main as gcs_main

from core.state_machine import FlightState
from utils.config_manager import ConfigManager


def test_live_start_requires_explicit_confirmation():
    result = gcs_main.main(["--mode", "live", "--start"])

    assert result == 2


def test_mock_wiring_check_runs_without_hardware():
    result = gcs_main.main(
        [
            "--mode",
            "mock",
            "--no-flight-recorder",
            "--perf-print-interval",
            "0",
        ]
    )

    assert result == 0


def test_mock_short_start_loop_runs_without_hardware():
    result = gcs_main.main(
        [
            "--mode",
            "mock",
            "--start",
            "--mock-fast",
            "--duration",
            "0.15",
            "--tick-rate-hz",
            "50",
            "--no-flight-recorder",
            "--perf-print-interval",
            "0",
        ]
    )

    assert result == 0


def test_build_controller_from_default_config():
    config = ConfigManager("config/default.yaml")
    controller = gcs_main.build_controller(config, "pickup_align")

    velocity = controller.compute_velocity(
        {"u": 320.0, "v": 240.0, "theta": 0.0},
        center_u=320.0,
        center_v=240.0,
        dt=0.1,
    )

    assert velocity == (0.0, 0.0, 0.0)


def test_state_needs_frame_only_for_vision_states():
    assert gcs_main.state_needs_frame(FlightState.TASK_REC_ALIGN)
    assert gcs_main.state_needs_frame(FlightState.TASK_REL_DESCEND)
    assert not gcs_main.state_needs_frame(FlightState.TRANS_DELIVERY)
    assert not gcs_main.state_needs_frame(FlightState.IDLE)
