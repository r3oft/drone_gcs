from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.servo_controller import VisualServoController
from core.state_machine import FlightState, GlobalFSM
from utils.config_manager import ConfigManager
from utils.logger import setup_logger
from utils.perf_monitor import PerfMonitor


VISION_STATES = {
    FlightState.TASK_REC_ALIGN,
    FlightState.TASK_REC_DESCEND,
    FlightState.TASK_REL_ALIGN,
    FlightState.TASK_REL_DESCEND,
}


class StopFlag:
    def __init__(self) -> None:
        self.requested = False


class SyntheticStreamer:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._opened = True

    def get_latest_frame(self) -> np.ndarray | None:
        if not self._opened:
            return None
        return self._frame.copy()

    def release(self) -> None:
        self._opened = False

    def is_opened(self) -> bool:
        return self._opened


class SyntheticTargetPoseEstimator:
    def __init__(
        self,
        center_u: float,
        center_v: float,
        behavior: str = "center",
    ) -> None:
        self._center_u = center_u
        self._center_v = center_v
        self._behavior = behavior

    def process_frame(self, frame: np.ndarray, target_cls_id: int) -> dict | None:
        if self._behavior == "lost" or target_cls_id < 0:
            return None
        return {
            "u": self._center_u,
            "v": self._center_v,
            "theta": 0.0,
            "conf": 1.0,
            "w": 64.0,
            "h": 64.0,
        }


@dataclass
class RuntimeComponents:
    config: ConfigManager
    streamer: Any
    perception: Any
    controller: VisualServoController
    flight: Any
    mcu: Any
    fsm: GlobalFSM
    perf: PerfMonitor


def cfg_float(config: ConfigManager, key: str, default: float) -> float:
    return float(config.get(key, default))


def cfg_int(config: ConfigManager, key: str, default: int) -> int:
    return int(config.get(key, default))


def cfg_bool(config: ConfigManager, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def option_or_config(value: Any, config: ConfigManager, key: str, default: Any) -> Any:
    return value if value is not None else config.get(key, default)


def build_controller(config: ConfigManager, profile: str) -> VisualServoController:
    base = f"servo.{profile}"
    return VisualServoController(
        kp=[
            cfg_float(config, f"{base}.kp.x", 0.003),
            cfg_float(config, f"{base}.kp.y", 0.003),
            cfg_float(config, f"{base}.kp.yaw", 0.6),
        ],
        kd=[
            cfg_float(config, f"{base}.kd.x", 0.001),
            cfg_float(config, f"{base}.kd.y", 0.001),
            cfg_float(config, f"{base}.kd.yaw", 0.1),
        ],
        deadband=[
            cfg_float(config, f"{base}.deadband.x", 30.0),
            cfg_float(config, f"{base}.deadband.y", 30.0),
            cfg_float(config, f"{base}.deadband.yaw", 0.1),
        ],
        max_vel=[
            cfg_float(config, f"{base}.max_vel.x", 0.3),
            cfg_float(config, f"{base}.max_vel.y", 0.3),
            cfg_float(config, f"{base}.max_vel.yaw", 0.5),
        ],
    )


def build_streamer(args: argparse.Namespace, config: ConfigManager) -> Any:
    if args.mode == "mock":
        if args.mock_source:
            from utils.mock import MockStreamer

            return MockStreamer(args.mock_source)
        return SyntheticStreamer(
            width=cfg_int(config, "camera.resolution.w", 640),
            height=cfg_int(config, "camera.resolution.h", 480),
        )

    from core.streamer import ZeroLatencyStreamer

    return ZeroLatencyStreamer(
        host=args.host or config.get("stream.host", "192.168.43.192"),
        stream_url=args.stream_url or config.get("stream.url"),
        capture_url=args.capture_url or config.get("stream.capture_url"),
        timeout_ms=int(
            option_or_config(args.timeout_ms, config, "stream.timeout_ms", 3000)
        ),
        prefer_stream=not args.prefer_capture
        and cfg_bool(config, "stream.prefer_stream", True),
        capture_poll_interval_s=cfg_float(
            config, "stream.capture_poll_interval_s", 1.0 / 15.0
        ),
        max_stale_s=cfg_float(config, "stream.max_stale_s", 2.0),
        stream_retry_interval_s=cfg_float(
            config, "stream.stream_retry_interval_s", 5.0
        ),
        stream_fail_threshold=cfg_int(config, "stream.stream_fail_threshold", 5),
        stream_preflight=args.stream_preflight,
    )


def build_perception(args: argparse.Namespace, config: ConfigManager) -> Any:
    if args.mode == "mock":
        return SyntheticTargetPoseEstimator(
            center_u=cfg_float(config, "camera.center_u", 320.0),
            center_v=cfg_float(config, "camera.center_v", 240.0),
            behavior=args.mock_target,
        )

    weights_path = Path(args.weights or config.get("perception.weights"))
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    from core.perception import TargetPoseEstimator

    return TargetPoseEstimator(
        weights_path=str(weights_path),
        conf_threshold=float(
            option_or_config(
                args.conf_threshold,
                config,
                "perception.conf_threshold",
                0.6,
            )
        ),
        device=args.device or config.get("perception.device", "cuda:0"),
    )


def build_flight_links(args: argparse.Namespace, config: ConfigManager) -> tuple[Any, Any]:
    if args.mode == "mock":
        from utils.mock import MockFlightBridge, MockMCUBridge

        flight = MockFlightBridge()
        mcu = MockMCUBridge()
        if args.mock_fast:
            from core.interfaces import MCUCommand, MCUResponse

            mcu.set_auto_response(MCUCommand.RESET, MCUResponse.RESET_DONE, 0.0)
            mcu.set_auto_response(MCUCommand.START_GRAB, MCUResponse.GRAB_DONE, 0.0)
            mcu.set_auto_response(
                MCUCommand.START_RELEASE, MCUResponse.RELEASE_DONE, 0.0
            )
        return flight, mcu

    from core.flight_bridge import FlightBridge, FlightConfig, MCUBridge

    flight_config = FlightConfig(
        connection_string=args.conn or config.get("mavlink.connection"),
        heartbeat_timeout=int(
            option_or_config(
                args.heartbeat_timeout,
                config,
                "mavlink.heartbeat_timeout_s",
                5,
            )
        ),
        takeoff_timeout_s=cfg_int(config, "flight.takeoff_timeout_s", 15),
        land_timeout_s=cfg_int(config, "flight.land_timeout_s", 20),
        land_detect_alt=cfg_float(config, "flight.land_detect_alt", 0.15),
        pixhawk_baud=int(option_or_config(args.baud, config, "mavlink.baud", 57600)),
        mcu_serial_port=cfg_int(config, "mcu.serial_port", 4),
        mcu_baudrate=cfg_int(config, "mcu.baudrate", 115200),
        reconnect_enabled=cfg_bool(config, "mavlink.reconnect_enabled", True),
        reconnect_max_attempts=cfg_int(config, "mavlink.reconnect_max_attempts", 3),
        reconnect_backoff_s=cfg_float(config, "mavlink.reconnect_backoff_s", 1.0),
    )
    flight = FlightBridge(flight_config)
    mcu = MCUBridge(flight, flight_config)
    return flight, mcu


def build_runtime(args: argparse.Namespace, logger: Any) -> RuntimeComponents:
    config = ConfigManager(args.config)
    overrides: dict[str, Any] = {}
    if args.tick_rate_hz is not None:
        overrides["fsm.tick_rate_hz"] = args.tick_rate_hz
    if args.no_flight_recorder:
        overrides["logging.enable_flight_recorder"] = False
    if overrides:
        config.override_from_args(overrides)

    perception = build_perception(args, config)
    controller = build_controller(config, args.servo_profile)
    streamer = build_streamer(args, config)
    flight, mcu = build_flight_links(args, config)
    fsm = GlobalFSM(flight, mcu, perception, controller, config)
    perf = PerfMonitor(
        window_size=cfg_int(config, "perf.window_size", 30),
        enable=cfg_bool(config, "perf.enable", True),
        warn_threshold_ms=cfg_float(config, "perf.warn_threshold_ms", 80.0),
    )
    logger.info("Runtime assembled in %s mode", args.mode)
    return RuntimeComponents(
        config=config,
        streamer=streamer,
        perception=perception,
        controller=controller,
        flight=flight,
        mcu=mcu,
        fsm=fsm,
        perf=perf,
    )


def state_needs_frame(state: FlightState) -> bool:
    return state in VISION_STATES


def run_control_loop(
    components: RuntimeComponents,
    args: argparse.Namespace,
    logger: Any,
    stop_flag: StopFlag,
) -> None:
    tick_rate_hz = float(
        args.tick_rate_hz
        if args.tick_rate_hz is not None
        else components.config.get("fsm.tick_rate_hz", 15)
    )
    if tick_rate_hz <= 0:
        raise ValueError("tick_rate_hz must be > 0")

    tick_interval_s = 1.0 / tick_rate_hz
    perf_print_interval = int(
        args.perf_print_interval
        if args.perf_print_interval is not None
        else components.config.get("perf.print_interval", 150)
    )
    started_at = time.monotonic()
    frame_count = 0

    while not stop_flag.requested:
        if args.duration > 0 and time.monotonic() - started_at >= args.duration:
            logger.info("Duration reached; stopping main loop")
            break

        loop_start = time.monotonic()
        frame = None

        with components.perf.measure("full_loop"):
            if state_needs_frame(components.fsm.state):
                with components.perf.measure("grab_frame"):
                    frame = components.streamer.get_latest_frame()

            with components.perf.measure("fsm_tick"):
                components.fsm.tick(frame)

        frame_count += 1
        if perf_print_interval > 0 and frame_count % perf_print_interval == 0:
            components.perf.print_summary(logger=logger)

        elapsed = time.monotonic() - loop_start
        sleep_s = max(0.0, tick_interval_s - elapsed)
        if sleep_s > 0:
            time.sleep(sleep_s)


def close_runtime(components: RuntimeComponents | None, logger: Any) -> None:
    if components is None:
        return

    recorder = getattr(components.fsm, "_recorder", None)
    if recorder is not None:
        recorder.close()

    release = getattr(components.streamer, "release", None)
    if callable(release):
        release()

    vehicle = getattr(components.flight, "_vehicle", None)
    if vehicle is not None:
        try:
            vehicle.close()
        except Exception as exc:
            logger.warning("Failed to close vehicle cleanly: %s", exc)


def install_signal_handlers(stop_flag: StopFlag, logger: Any) -> None:
    def _handle_signal(signum, _frame) -> None:
        logger.warning("Signal %s received; stopping main loop", signum)
        stop_flag.requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the drone GCS main loop")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    parser.add_argument("--start", action="store_true", help="request FSM start")
    parser.add_argument(
        "--confirm-live-start",
        action="store_true",
        help="required with --mode live --start",
    )
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--hold-idle", action="store_true")
    parser.add_argument("--tick-rate-hz", type=float, default=None)
    parser.add_argument("--perf-print-interval", type=int, default=None)
    parser.add_argument("--no-flight-recorder", action="store_true")

    parser.add_argument("--servo-profile", default="pickup_align")
    parser.add_argument("--mock-source", default=None)
    parser.add_argument("--mock-target", choices=("center", "lost"), default="center")
    parser.add_argument("--mock-fast", action="store_true")

    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf-threshold", type=float, default=None)

    parser.add_argument("--host", default=None)
    parser.add_argument("--stream-url", default=None)
    parser.add_argument("--capture-url", default=None)
    parser.add_argument("--prefer-capture", action="store_true")
    parser.add_argument("--stream-preflight", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=None)

    parser.add_argument("--conn", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--heartbeat-timeout", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logger("Main")

    if args.mode == "live" and args.start and not args.confirm_live_start:
        logger.error(
            "Refusing live auto-start without --confirm-live-start. "
            "Use mock mode for desktop integration tests."
        )
        return 2

    stop_flag = StopFlag()
    install_signal_handlers(stop_flag, logger)
    components: RuntimeComponents | None = None

    try:
        components = build_runtime(args, logger)

        if args.start:
            components.fsm.request_start()
            logger.info("Start requested; entering control loop")
        elif not args.hold_idle and args.duration <= 0:
            components.fsm.tick(None)
            logger.info("Wiring check complete. Pass --start to run the mission.")
            return 0

        run_control_loop(components, args, logger, stop_flag)
        return 0

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt; requesting stop")
        if components is not None:
            components.fsm.request_stop()
        return 130
    except Exception as exc:
        logger.exception("Main loop failed: %s", exc)
        if components is not None:
            components.fsm.request_stop()
        return 1
    finally:
        close_runtime(components, logger)


if __name__ == "__main__":
    raise SystemExit(main())
