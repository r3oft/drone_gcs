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


MISSION_STANDARD = "standard"
MISSION_NO_MCU_FLIGHT_TEST = "no_mcu_flight_test"

VISION_STATES = {
    FlightState.TASK_REC_ALIGN,
    FlightState.TASK_REC_DESCEND,
    FlightState.TASK_REL_ALIGN,
    FlightState.TASK_REL_DESCEND,
}

TARGET_DEFAULT_CLS_IDS = {
    "pickup_zone": 0,
    "delivery_zone": 1,
}


def control_center(config: ConfigManager, frame_shape: tuple[int, ...] | None = None) -> tuple[float, float]:
    fallback_u = frame_shape[1] / 2 if frame_shape is not None and len(frame_shape) >= 2 else 160.0
    fallback_v = frame_shape[0] / 2 if frame_shape is not None and len(frame_shape) >= 2 else 120.0
    center_u = cfg_float(config, "camera.center_u", fallback_u)
    center_v = cfg_float(config, "camera.center_v", fallback_v)
    center_u += cfg_float(config, "camera.control_center_offset_u_px", 0.0)
    center_v += cfg_float(config, "camera.control_center_offset_v_px", 0.0)
    return center_u, center_v


class StopFlag:
    def __init__(self) -> None:
        self.requested = False


class SyntheticStreamer:
    def __init__(self, width: int = 320, height: int = 240) -> None:
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


class NullMCUBridge:
    def __init__(self) -> None:
        self._pending_response: str | None = None
        self.commands: list[str] = []

    def connect(self) -> bool:
        return True

    def send_command(self, command: str) -> bool:
        self.commands.append(command)
        if command == "RESET":
            self._pending_response = "RESET_DONE"
        return True

    def get_latest_response(self) -> str | None:
        response = self._pending_response
        self._pending_response = None
        return response

    def is_connected(self) -> bool:
        return True

    def close(self) -> None:
        self._pending_response = None


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


def split_record_paths(record_path: str | None) -> tuple[str | None, str | None]:
    """Return (with_hud_path, non_hud_path) under the requested record directory."""
    if record_path is None:
        return None, None

    path = Path(record_path)
    return (
        str(path.parent / "with_hud" / path.name),
        str(path.parent / "non_hud" / path.name),
    )


def option_or_config(value: Any, config: ConfigManager, key: str, default: Any) -> Any:
    return value if value is not None else config.get(key, default)


def get_mission_profile(args: argparse.Namespace, config: ConfigManager) -> str:
    return str(
        option_or_config(args.mission_profile, config, "mission.profile", MISSION_STANDARD)
    )


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
        axis_map=str(config.get(f"{base}.axis_map", "standard")),
        axis_sign=[
            cfg_float(config, f"{base}.axis_sign.x", 1.0),
            cfg_float(config, f"{base}.axis_sign.y", 1.0),
            cfg_float(config, f"{base}.axis_sign.yaw", 1.0),
        ],
    )


def build_streamer(args: argparse.Namespace, config: ConfigManager) -> Any:
    if args.mode == "mock":
        if args.mock_source:
            from utils.mock import MockStreamer

            return MockStreamer(args.mock_source)
        return SyntheticStreamer(
            width=cfg_int(config, "camera.resolution.w", 320),
            height=cfg_int(config, "camera.resolution.h", 240),
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
        center_u, center_v = control_center(config)
        return SyntheticTargetPoseEstimator(
            center_u=center_u,
            center_v=center_v,
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
    mission_profile = get_mission_profile(args, config)

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

    from core.flight_bridge import FlightBridge, FlightConfig

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
        goto_vertical_speed=cfg_float(config, "flight.goto_vertical_speed", 0.15),
        goto_alt_tolerance=cfg_float(config, "flight.goto_alt_tolerance", 0.05),
        goto_command_hz=cfg_float(config, "flight.goto_command_hz", 5.0),
        pixhawk_baud=int(option_or_config(args.baud, config, "mavlink.baud", 57600)),
        mcu_serial_port=cfg_int(
            config,
            "mcu.pixhawk_serial_port",
            cfg_int(config, "mcu.serial_port", 4),
        ),
        mcu_baudrate=cfg_int(config, "mcu.baudrate", 115200),
        reconnect_enabled=cfg_bool(config, "mavlink.reconnect_enabled", True),
        reconnect_max_attempts=cfg_int(config, "mavlink.reconnect_max_attempts", 3),
        reconnect_backoff_s=cfg_float(config, "mavlink.reconnect_backoff_s", 1.0),
    )
    flight = FlightBridge(flight_config)

    mcu_transport = option_or_config(
        args.mcu_transport,
        config,
        "mcu.transport",
        "direct_serial",
    )
    if mcu_transport == "direct_serial":
        from core.mcu_bridge import DirectSerialMCUBridge, MCUConfig

        mcu_config = MCUConfig(
            port=str(
                option_or_config(args.mcu_port, config, "mcu.port", "/dev/ttyACM0")
            ),
            baudrate=int(
                option_or_config(args.mcu_baud, config, "mcu.baudrate", 57600)
            ),
            read_timeout_s=cfg_float(config, "mcu.read_timeout_s", 0.02),
            write_timeout_s=cfg_float(config, "mcu.write_timeout_s", 0.5),
        )
        mcu = DirectSerialMCUBridge(mcu_config)
    elif mcu_transport == "pixhawk_serial_control":
        from core.flight_bridge import MCUBridge

        mcu = MCUBridge(flight, flight_config)
    else:
        raise ValueError(f"Unsupported MCU transport: {mcu_transport}")
    return flight, mcu


def build_runtime(args: argparse.Namespace, logger: Any) -> RuntimeComponents:
    config = ConfigManager(args.config)
    overrides: dict[str, Any] = {}
    if args.tick_rate_hz is not None:
        overrides["fsm.tick_rate_hz"] = args.tick_rate_hz
    if args.reset_timeout is not None:
        overrides["fsm.reset_timeout_s"] = args.reset_timeout
    if args.takeoff_timeout is not None:
        overrides["flight.takeoff_timeout_s"] = args.takeoff_timeout
    if args.takeoff_alt_tolerance is not None:
        overrides["flight.takeoff_alt_tolerance_m"] = args.takeoff_alt_tolerance
    if args.mission_profile is not None:
        overrides["mission.profile"] = args.mission_profile
    if args.camera_link_loss_timeout is not None:
        overrides["mission.camera_link_loss_timeout_s"] = (
            args.camera_link_loss_timeout
        )
    if args.max_flight_alt is not None:
        overrides["mission.max_flight_alt_m"] = args.max_flight_alt
    if args.pre_pickup_forward is not None:
        overrides["mission.pre_pickup_forward_m"] = args.pre_pickup_forward
    if args.pre_pickup_speed is not None:
        overrides["mission.pre_pickup_speed_mps"] = args.pre_pickup_speed
    if args.no_mcu_pickup_use_land_mode:
        overrides["mission.no_mcu_pickup_use_land_mode"] = True
    if args.no_mcu_pickup_touchdown_alt is not None:
        overrides["mission.no_mcu_pickup_touchdown_alt_m"] = args.no_mcu_pickup_touchdown_alt
    if args.no_mcu_pickup_touchdown_hold is not None:
        overrides["mission.no_mcu_pickup_touchdown_hold_s"] = args.no_mcu_pickup_touchdown_hold
    if args.no_mcu_pickup_min_descend is not None:
        overrides["mission.no_mcu_pickup_min_descend_s"] = args.no_mcu_pickup_min_descend
    if args.no_mcu_pickup_descend_vz is not None:
        overrides["mission.no_mcu_pickup_descend_vz_mps"] = args.no_mcu_pickup_descend_vz
    if args.no_mcu_pickup_descend_timeout is not None:
        overrides["mission.no_mcu_pickup_descend_timeout_s"] = args.no_mcu_pickup_descend_timeout
    if args.no_mcu_retakeoff_climb_vz is not None:
        overrides["mission.no_mcu_retakeoff_climb_vz_mps"] = args.no_mcu_retakeoff_climb_vz
    if args.no_mcu_enable_yaw_align:
        overrides["mission.no_mcu_ignore_yaw_alignment"] = False
    if args.no_mcu_ignore_yaw_align:
        overrides["mission.no_mcu_ignore_yaw_alignment"] = True
    if args.no_mcu_enable_yaw_align and args.no_mcu_ignore_yaw_align:
        raise ValueError(
            "--no-mcu-enable-yaw-align and --no-mcu-ignore-yaw-align are mutually exclusive"
        )
    if args.no_mcu_retakeoff_delay is not None:
        overrides["mission.no_mcu_retakeoff_delay_s"] = args.no_mcu_retakeoff_delay
    if args.no_mcu_retakeoff_wait_timeout is not None:
        overrides["mission.no_mcu_retakeoff_wait_timeout_s"] = (
            args.no_mcu_retakeoff_wait_timeout
        )
    if args.allow_retakeoff_while_armed:
        overrides["mission.no_mcu_retakeoff_require_disarmed"] = False
    if args.servo_axis_map is not None:
        overrides["servo.pickup_align.axis_map"] = args.servo_axis_map
        overrides["servo.delivery_align.axis_map"] = args.servo_axis_map
    for axis, value in (
        ("x", args.servo_sign_x),
        ("y", args.servo_sign_y),
        ("yaw", args.servo_sign_yaw),
    ):
        if value is not None:
            overrides[f"servo.pickup_align.axis_sign.{axis}"] = value
            overrides[f"servo.delivery_align.axis_sign.{axis}"] = value
    if args.servo_max_yaw is not None:
        overrides["servo.pickup_align.max_vel.yaw"] = args.servo_max_yaw
        overrides["servo.delivery_align.max_vel.yaw"] = args.servo_max_yaw
    if args.camera_center_offset_u is not None:
        overrides["camera.control_center_offset_u_px"] = args.camera_center_offset_u
    if args.camera_center_offset_v is not None:
        overrides["camera.control_center_offset_v_px"] = args.camera_center_offset_v
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


def opencv_has_gui() -> bool:
    try:
        import cv2
    except Exception:
        return False
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return "NONE" not in line.upper()
    return True


def safe_get_telemetry(flight: Any) -> dict[str, Any]:
    try:
        telemetry = dict(flight.get_telemetry())
    except Exception:
        telemetry = {}

    get_local_altitude = getattr(flight, "_get_local_altitude", None)
    if callable(get_local_altitude):
        try:
            local_alt = get_local_altitude()
            if local_alt is not None:
                telemetry["local_alt"] = float(local_alt)
        except Exception:
            pass

    latest_local_position_time = getattr(flight, "_latest_local_position_time", 0.0)
    if latest_local_position_time:
        telemetry["local_position_age_s"] = max(
            0.0, time.time() - float(latest_local_position_time)
        )
    return telemetry


def target_name_for_state(state: FlightState) -> str:
    if state in (FlightState.TASK_REC_ALIGN, FlightState.TASK_REC_DESCEND):
        return "pickup_zone"
    if state in (FlightState.TASK_REL_ALIGN, FlightState.TASK_REL_DESCEND):
        return "delivery_zone"
    return "none"


def debug_target_name_for_state(state: FlightState) -> str:
    if state in (
        FlightState.TRANS_DELIVERY,
        FlightState.TASK_REL_ALIGN,
        FlightState.TASK_REL_DESCEND,
        FlightState.TASK_REL_RELEASE,
        FlightState.OUTBOUND,
    ):
        return "delivery_zone"
    return "pickup_zone"


def target_cls_id(config: ConfigManager, target_name: str) -> int:
    return int(
        config.get(
            f"perception.task_targets.{target_name}.cls_id",
            TARGET_DEFAULT_CLS_IDS[target_name],
        )
    )


class MainDebugHUD:
    def __init__(
        self,
        args: argparse.Namespace,
        config: ConfigManager,
        logger: Any,
    ) -> None:
        self.enabled = bool(args.debug_hud or args.debug_hud_record_path)
        self.display_enabled = bool(args.debug_hud)
        self._window_name = args.debug_window_name
        self._logger = logger
        self._config = config
        self._cv2 = None
        self._visualizer = None
        self._raw_visualizer = None
        self._colors: dict[str, tuple[int, int, int]] = {}
        self._target_mode = args.debug_hud_target
        self._debug_controllers: dict[str, VisualServoController] = {}
        self._preview_error_logged = False
        self._record_segment_s = float(args.debug_hud_record_segment_s)
        self._record_flush_on_gap_s = float(args.debug_hud_record_flush_on_gap_s)
        self._last_record_write_time = time.monotonic()
        self._recording_flushed_for_gap = False
        self.last_target_name = "none"
        self.last_target_found = False
        self.last_target_source = "none"

        if not self.enabled:
            return

        import cv2
        from utils.visualization import CLASS_COLORS, DebugVisualizer

        self._cv2 = cv2
        self._colors = CLASS_COLORS
        with_hud_record_path, non_hud_record_path = split_record_paths(
            args.debug_hud_record_path
        )
        self._raw_visualizer = DebugVisualizer(
            record_path=non_hud_record_path,
            record_segment_s=self._record_segment_s,
        )
        self._visualizer = DebugVisualizer(
            record_path=with_hud_record_path,
            hud_corner=args.debug_hud_corner,
            record_segment_s=self._record_segment_s,
        )
        if args.debug_hud_record_path:
            logger.info(
                "Debug HUD recording enabled: with_hud=%s non_hud=%s",
                with_hud_record_path,
                non_hud_record_path,
            )
        self._debug_controllers = {
            "pickup_zone": build_controller(config, "pickup_align"),
            "delivery_zone": build_controller(config, "delivery_align"),
        }
        import os

        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if self.display_enabled and not opencv_has_gui():
            logger.warning("OpenCV GUI is unavailable; disabling --debug-hud display")
            self.display_enabled = False
        if self.display_enabled and not has_display:
            logger.warning("No DISPLAY/WAYLAND_DISPLAY found; disabling --debug-hud display")
            self.display_enabled = False

    def _select_target_name(self, state: FlightState) -> str:
        if self._target_mode != "auto":
            return self._target_mode
        return debug_target_name_for_state(state)

    def _preview_detection(
        self,
        frame: np.ndarray,
        components: RuntimeComponents,
        state: FlightState,
    ) -> tuple[str, dict | None, dict | None, tuple[float, float, float]]:
        target_name = self._select_target_name(state)
        try:
            target = components.perception.process_frame(
                frame,
                target_cls_id(components.config, target_name),
            )
        except Exception as exc:
            if not self._preview_error_logged:
                self._logger.warning("Debug HUD preview inference failed: %s", exc)
                self._preview_error_logged = True
            return target_name, None, None, (0.0, 0.0, 0.0)

        if target is None:
            return target_name, None, None, (0.0, 0.0, 0.0)

        controller = self._debug_controllers[target_name]
        center_u, center_v = control_center(components.config, frame.shape)
        debug = controller.compute_debug(
            target,
            center_u=center_u,
            center_v=center_v,
            dt=0.0,
        )
        return target_name, target, debug, debug["velocities"]

    def update(
        self,
        frame: np.ndarray | None,
        components: RuntimeComponents,
        source_fps: float,
        read_fps: float,
    ) -> bool:
        if not self.enabled:
            return True

        if frame is None:
            now = time.monotonic()
            if (
                self._record_segment_s > 0
                and self._record_flush_on_gap_s > 0
                and not self._recording_flushed_for_gap
                and now - self._last_record_write_time >= self._record_flush_on_gap_s
            ):
                if self._raw_visualizer is not None:
                    self._raw_visualizer.flush_recording()
                if self._visualizer is not None:
                    self._visualizer.flush_recording()
                self._recording_flushed_for_gap = True
            return True

        if self._visualizer is None:
            return True

        cv2 = self._cv2
        if cv2 is None:
            return True

        annotated = frame.copy()
        fsm = components.fsm
        state = fsm.state
        target = getattr(fsm, "_tick_target", None)
        debug = getattr(fsm, "_tick_debug", None)
        vel = getattr(fsm, "_tick_vel", None) or (0.0, 0.0, 0.0)
        tel = safe_get_telemetry(components.flight)
        label = target_name_for_state(state)
        target_source = "fsm"
        if target is None:
            label, target, debug, vel = self._preview_detection(frame, components, state)
            target_source = "preview" if target is not None else "none"
        self.last_target_name = label if label != "none" else self._select_target_name(state)
        self.last_target_found = target is not None
        self.last_target_source = target_source

        center_u, center_v = control_center(components.config, annotated.shape)
        center = (int(round(center_u)), int(round(center_v)))
        cv2.drawMarker(
            annotated,
            center,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            16,
            1,
        )

        if target is not None:
            color = self._colors.get(label, self._colors.get("default", (0, 200, 255)))
            display_label = f"{label} {target_source}" if target_source == "preview" else label
            self._visualizer.draw_obb(
                annotated,
                u=target["u"],
                v=target["v"],
                w=target.get("w", 30.0),
                h=target.get("h", 30.0),
                theta=target.get("theta", 0.0),
                label=display_label,
                conf=target.get("conf", 0.0),
                color=color,
                thickness=3,
            )
            self._visualizer.draw_error_vector(
                annotated,
                center=center,
                target=(int(round(target["u"])), int(round(target["v"]))),
                color=color,
            )

        vx, vy, vyaw = vel
        hud = {
            "state": state.name,
            "camera": getattr(components.streamer, "current_mode", "mock"),
            "mode": tel.get("mode", "unknown"),
            "target": label if target is not None else f"{self._select_target_name(state)} none",
            "source_fps": source_fps,
            "read_fps": read_fps,
            "fps": source_fps,
            "vx": vx,
            "vy": vy,
            "vyaw": vyaw,
            "alt": float(tel.get("alt", 0.0) or 0.0),
            "armed": bool(tel.get("armed", False)),
            "heartbeat_ok": bool(tel.get("heartbeat_ok", False)),
        }
        if "local_alt" in tel:
            hud["local_alt"] = float(tel["local_alt"])
        if target is not None and "conf" in target:
            hud["conf"] = target["conf"]
        if debug is not None and target is not None:
            err_x, err_y, err_yaw = debug["errors"]
            p_x, p_y, p_yaw = debug["p_terms"]
            d_x, d_y, d_yaw = debug["d_terms"]
            hud.update(
                {
                    "dt": debug["dt"],
                    "err_x": err_x,
                    "err_y": err_y,
                    "err_yaw": err_yaw,
                    "p_x": p_x,
                    "p_y": p_y,
                    "p_yaw": p_yaw,
                    "d_x": d_x,
                    "d_y": d_y,
                    "d_yaw": d_yaw,
                }
            )

        self._visualizer.draw_hud(annotated, hud)
        if self._raw_visualizer is not None:
            self._raw_visualizer.write_frame(frame)
        self._visualizer.write_frame(annotated)
        self._last_record_write_time = time.monotonic()
        self._recording_flushed_for_gap = False

        if self.display_enabled:
            cv2.imshow(self._window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self._logger.info("Debug HUD requested stop")
                return False
        return True

    def release(self) -> None:
        if self._raw_visualizer is not None:
            self._raw_visualizer.release()
        if self._visualizer is not None:
            self._visualizer.release()
        if self.display_enabled and self._cv2 is not None:
            self._cv2.destroyAllWindows()


def log_runtime_status(
    logger: Any,
    components: RuntimeComponents,
    frame_requested: bool,
    frame_available: bool,
    source_fps: float,
    read_fps: float,
    debug_hud: MainDebugHUD | None = None,
) -> None:
    fsm = components.fsm
    tel = safe_get_telemetry(components.flight)
    target = getattr(fsm, "_tick_target", None)
    debug = getattr(fsm, "_tick_debug", None)
    vel = getattr(fsm, "_tick_vel", None)
    target_label = target_name_for_state(fsm.state) if target is not None else "none"

    parts = [
        f"state={fsm.state.name}",
        f"mode={tel.get('mode', 'unknown')}",
        f"armed={tel.get('armed', 'unknown')}",
        f"alt={float(tel.get('alt', 0.0) or 0.0):.2f}",
        f"local_alt={float(tel.get('local_alt', 0.0) or 0.0):.2f}",
        f"local_age={float(tel.get('local_position_age_s', -1.0)):.2f}",
        f"frame={'yes' if frame_available else 'none' if frame_requested else 'skip'}",
        f"stream={getattr(components.streamer, 'current_mode', 'mock')}",
        f"src_fps={source_fps:.1f}",
        f"read_fps={read_fps:.1f}",
        f"target={target_label}",
    ]
    if debug_hud is not None and debug_hud.enabled:
        parts.append(
            f"hud_target={debug_hud.last_target_name}:"
            f"{'yes' if debug_hud.last_target_found else 'no'}:"
            f"{debug_hud.last_target_source}"
        )
    if vel is not None:
        vx, vy, vyaw = vel
        parts.append(f"vel=({vx:+.3f},{vy:+.3f},{vyaw:+.3f})")
    if debug is not None:
        err_x, err_y, err_yaw = debug["errors"]
        parts.append(f"err=({err_x:+.1f},{err_y:+.1f},{err_yaw:+.2f})")
        p_x, p_y, p_yaw = debug.get("p_terms", (0.0, 0.0, 0.0))
        d_x, d_y, d_yaw = debug.get("d_terms", (0.0, 0.0, 0.0))
        parts.append(f"p=({p_x:+.3f},{p_y:+.3f},{p_yaw:+.3f})")
        parts.append(f"d=({d_x:+.3f},{d_y:+.3f},{d_yaw:+.3f})")
    logger.info("Status — %s", " ".join(parts))


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
    last_status_log_time = started_at
    last_fps_time = started_at
    last_update_count = int(getattr(components.streamer, "frame_update_count", 0))
    interval_reads = 0
    source_fps = 0.0
    read_fps = 0.0
    status_log_interval = float(args.status_log_interval)
    debug_hud = MainDebugHUD(args, components.config, logger)

    try:
        while not stop_flag.requested:
            now_mono = time.monotonic()
            if args.duration > 0 and now_mono - started_at >= args.duration:
                logger.info("Duration reached; stopping main loop")
                break

            loop_start = time.monotonic()
            frame = None
            frame_for_fsm = None
            frame_requested_for_fsm = state_needs_frame(components.fsm.state)
            frame_requested = frame_requested_for_fsm or debug_hud.enabled

            with components.perf.measure("full_loop"):
                if frame_requested:
                    with components.perf.measure("grab_frame"):
                        frame = components.streamer.get_latest_frame()
                    if frame is not None:
                        interval_reads += 1
                    if frame_requested_for_fsm:
                        frame_for_fsm = frame

                with components.perf.measure("fsm_tick"):
                    components.fsm.tick(frame_for_fsm)

            frame_count += 1

            now_mono = time.monotonic()
            fps_dt = now_mono - last_fps_time
            if fps_dt >= 1.0:
                update_count = int(getattr(components.streamer, "frame_update_count", 0))
                if update_count >= last_update_count:
                    source_fps = (update_count - last_update_count) / fps_dt
                else:
                    source_fps = 0.0
                read_fps = interval_reads / fps_dt
                last_update_count = update_count
                interval_reads = 0
                last_fps_time = now_mono

            if not debug_hud.update(frame, components, source_fps, read_fps):
                stop_flag.requested = True

            if (
                status_log_interval > 0
                and now_mono - last_status_log_time >= status_log_interval
            ):
                log_runtime_status(
                    logger,
                    components,
                    frame_requested=frame_requested,
                    frame_available=frame is not None,
                    source_fps=source_fps,
                    read_fps=read_fps,
                    debug_hud=debug_hud,
                )
                last_status_log_time = now_mono

            if perf_print_interval > 0 and frame_count % perf_print_interval == 0:
                components.perf.print_summary(logger=logger)

            elapsed = time.monotonic() - loop_start
            sleep_s = max(0.0, tick_interval_s - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        debug_hud.release()


def close_runtime(components: RuntimeComponents | None, logger: Any) -> None:
    if components is None:
        return

    stop_velocity = getattr(components.flight, "send_body_velocity", None)
    if callable(stop_velocity):
        try:
            stop_velocity(0, 0, 0, 0)
        except Exception as exc:
            logger.warning("Failed to send stop velocity during shutdown: %s", exc)

    recorder = getattr(components.fsm, "_recorder", None)
    if recorder is not None:
        recorder.close()

    release = getattr(components.streamer, "release", None)
    if callable(release):
        release()

    close_mcu = getattr(components.mcu, "close", None)
    if callable(close_mcu):
        try:
            close_mcu()
        except Exception as exc:
            logger.warning("Failed to close MCU link cleanly: %s", exc)

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
        raise KeyboardInterrupt

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
    parser.add_argument("--reset-timeout", type=float, default=None)
    parser.add_argument("--takeoff-timeout", type=float, default=None)
    parser.add_argument("--takeoff-alt-tolerance", type=float, default=None)
    parser.add_argument("--perf-print-interval", type=int, default=None)
    parser.add_argument("--status-log-interval", type=float, default=0.0)
    parser.add_argument("--no-flight-recorder", action="store_true")
    parser.add_argument(
        "--mission-profile",
        choices=(MISSION_STANDARD, MISSION_NO_MCU_FLIGHT_TEST),
        default=None,
    )
    parser.add_argument("--camera-link-loss-timeout", type=float, default=None)
    parser.add_argument("--max-flight-alt", type=float, default=None)
    parser.add_argument("--pre-pickup-forward", type=float, default=None)
    parser.add_argument("--pre-pickup-speed", type=float, default=None)
    parser.add_argument("--no-mcu-pickup-use-land-mode", action="store_true")
    parser.add_argument("--no-mcu-pickup-touchdown-alt", type=float, default=None)
    parser.add_argument("--no-mcu-pickup-touchdown-hold", type=float, default=None)
    parser.add_argument("--no-mcu-pickup-min-descend", type=float, default=None)
    parser.add_argument("--no-mcu-pickup-descend-vz", type=float, default=None)
    parser.add_argument("--no-mcu-pickup-descend-timeout", type=float, default=None)
    parser.add_argument("--no-mcu-retakeoff-climb-vz", type=float, default=None)
    parser.add_argument("--no-mcu-enable-yaw-align", action="store_true")
    parser.add_argument("--no-mcu-ignore-yaw-align", action="store_true")
    parser.add_argument("--no-mcu-retakeoff-delay", type=float, default=None)
    parser.add_argument("--no-mcu-retakeoff-wait-timeout", type=float, default=None)
    parser.add_argument(
        "--allow-retakeoff-while-armed",
        action="store_true",
        help="do not require disarm before the no-MCU second takeoff",
    )
    parser.add_argument(
        "--servo-axis-map",
        choices=("standard", "swap_xy"),
        default=None,
        help="override servo pickup/delivery image-to-body axis mapping",
    )
    parser.add_argument("--servo-sign-x", type=float, choices=(-1.0, 1.0), default=None)
    parser.add_argument("--servo-sign-y", type=float, choices=(-1.0, 1.0), default=None)
    parser.add_argument("--servo-sign-yaw", type=float, choices=(-1.0, 1.0), default=None)
    parser.add_argument("--servo-max-yaw", type=float, default=None)
    parser.add_argument("--camera-center-offset-u", type=float, default=None)
    parser.add_argument("--camera-center-offset-v", type=float, default=None)
    parser.add_argument("--debug-hud", action="store_true")
    parser.add_argument("--debug-window-name", default="Drone GCS Main HUD")
    parser.add_argument(
        "--debug-hud-target",
        choices=("auto", "pickup_zone", "delivery_zone"),
        default="auto",
    )
    parser.add_argument("--debug-hud-record-path", "--record-path", default=None)
    parser.add_argument(
        "--debug-hud-record-segment-s",
        type=float,
        default=0.0,
        help="split HUD recordings into finalized AVI segments; 0 keeps one file",
    )
    parser.add_argument(
        "--debug-hud-record-flush-on-gap-s",
        type=float,
        default=2.0,
        help="with segmented recording, finalize current segment after this many seconds without frames",
    )
    parser.add_argument(
        "--debug-hud-corner",
        choices=("top_left", "top_right", "bottom_left", "bottom_right"),
        default="top_left",
    )

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
    parser.add_argument("--mcu-port", default=None)
    parser.add_argument("--mcu-baud", type=int, default=None)
    parser.add_argument(
        "--mcu-transport",
        choices=("direct_serial", "pixhawk_serial_control"),
        default=None,
    )
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
