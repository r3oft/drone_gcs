import sys
import types

import main as gcs_main

from core.state_machine import FlightState
from utils.config_manager import ConfigManager


def install_fake_live_modules(monkeypatch):
    class FakeFlightConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeFlightBridge:
        def __init__(self, config):
            self.config = config

    class FakeLegacyMCUBridge:
        def __init__(self, flight, config):
            self.flight = flight
            self.config = config

    class FakeMCUConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeDirectSerialMCUBridge:
        def __init__(self, config):
            self.config = config

    monkeypatch.setitem(
        sys.modules,
        "core.flight_bridge",
        types.SimpleNamespace(
            FlightConfig=FakeFlightConfig,
            FlightBridge=FakeFlightBridge,
            MCUBridge=FakeLegacyMCUBridge,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "core.mcu_bridge",
        types.SimpleNamespace(
            MCUConfig=FakeMCUConfig,
            DirectSerialMCUBridge=FakeDirectSerialMCUBridge,
        ),
    )
    return types.SimpleNamespace(
        FlightBridge=FakeFlightBridge,
        LegacyMCUBridge=FakeLegacyMCUBridge,
        DirectSerialMCUBridge=FakeDirectSerialMCUBridge,
    )


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


def test_live_build_uses_direct_serial_mcu_by_default(monkeypatch):
    classes = install_fake_live_modules(monkeypatch)
    args = gcs_main.parse_args(
        [
            "--mode",
            "live",
            "--mcu-port",
            "loop://",
            "--mcu-baud",
            "57600",
        ]
    )
    config = ConfigManager("config/default.yaml")

    flight, mcu = gcs_main.build_flight_links(args, config)

    assert isinstance(flight, classes.FlightBridge)
    assert isinstance(mcu, classes.DirectSerialMCUBridge)
    assert mcu.config.port == "loop://"
    assert mcu.config.baudrate == 57600
    assert mcu.config.read_timeout_s == 0.02
    assert mcu.config.write_timeout_s == 0.5


def test_live_build_can_use_legacy_pixhawk_mcu_transport(monkeypatch):
    classes = install_fake_live_modules(monkeypatch)
    args = gcs_main.parse_args(
        [
            "--mode",
            "live",
            "--mcu-transport",
            "pixhawk_serial_control",
        ]
    )
    config = ConfigManager("config/default.yaml")

    flight, mcu = gcs_main.build_flight_links(args, config)

    assert isinstance(flight, classes.FlightBridge)
    assert isinstance(mcu, classes.LegacyMCUBridge)
    assert mcu.flight is flight
