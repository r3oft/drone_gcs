"""
Live visual HUD runner for ESP32-S3 stream + YOLO-OBB + PD debug output.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, ".")

from core.servo_controller import VisualServoController
from core.streamer import ZeroLatencyStreamer
from utils.config_manager import ConfigManager
from utils.visualization import CLASS_COLORS, DebugVisualizer


TARGETS = {
    "pickup_zone": 0,
    "delivery_zone": 1,
}


def opencv_has_gui() -> bool:
    """Return False for opencv-python-headless builds."""
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return "NONE" not in line.upper()
    return True


def cfg_float(config: ConfigManager, key: str, default: float) -> float:
    return float(config.get(key, default))


def cfg_bool(config: ConfigManager, key: str, default: bool) -> bool:
    return bool(config.get(key, default))


def make_controller(config: ConfigManager, profile: str) -> VisualServoController:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live visual HUD")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--weights", default="weights/cargo_obb_run2/weights/best.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf-threshold", type=float, default=None)

    parser.add_argument("--target", choices=sorted(TARGETS), default="pickup_zone")
    parser.add_argument("--servo-profile", choices=("pickup_align", "delivery_align"), default=None)
    parser.add_argument("--infer-hz", type=float, default=15.0)
    parser.add_argument("--output-hz", type=float, default=15.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 means run until interrupted")

    parser.add_argument("--host", default=None)
    parser.add_argument("--stream-url", default=None)
    parser.add_argument("--capture-url", default=None)
    parser.add_argument("--prefer-capture", action="store_true")
    parser.add_argument("--stream-preflight", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=None)

    parser.add_argument("--display", action="store_true", help="Use cv2.imshow when GUI OpenCV is available")
    parser.add_argument("--window-name", default="Drone GCS Visual HUD")
    parser.add_argument("--record-path", default=None)
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--snapshot-every-s", type=float, default=0.0)
    parser.add_argument("--report-interval-s", type=float, default=1.0)
    parser.add_argument("--detection-stale-s", type=float, default=0.5)

    parser.add_argument("--center-u", type=float, default=None)
    parser.add_argument("--center-v", type=float, default=None)
    parser.add_argument(
        "--use-config-center",
        action="store_true",
        help="Use camera.center_u/v from config instead of the current frame center",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConfigManager(args.config)

    from core.perception import TargetPoseEstimator

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    device = args.device or config.get("perception.device", "cuda:0")
    conf_threshold = (
        args.conf_threshold
        if args.conf_threshold is not None
        else cfg_float(config, "perception.conf_threshold", 0.6)
    )
    target_cls_id = int(
        config.get(f"perception.task_targets.{args.target}.cls_id", TARGETS[args.target])
    )
    servo_profile = args.servo_profile
    if servo_profile is None:
        servo_profile = "pickup_align" if args.target == "pickup_zone" else "delivery_align"

    display_enabled = bool(args.display)
    if display_enabled and not opencv_has_gui():
        print("WARN: OpenCV was built without GUI support; disabling --display.")
        display_enabled = False

    print("=== Visual HUD runner ===")
    print(f"  target          : {args.target} (cls_id={target_cls_id})")
    print(f"  weights         : {weights_path}")
    print(f"  device          : {device}")
    print(f"  conf_threshold  : {conf_threshold}")
    print(f"  servo_profile   : {servo_profile}")
    print(f"  display         : {display_enabled}")
    print(f"  record_path     : {args.record_path}")
    print()

    streamer = ZeroLatencyStreamer(
        host=args.host or config.get("stream.host", "192.168.43.192"),
        stream_url=args.stream_url or config.get("stream.url"),
        capture_url=args.capture_url or config.get("stream.capture_url"),
        timeout_ms=args.timeout_ms or int(config.get("stream.timeout_ms", 3000)),
        prefer_stream=not args.prefer_capture and cfg_bool(config, "stream.prefer_stream", True),
        capture_poll_interval_s=cfg_float(config, "stream.capture_poll_interval_s", 1.0 / 15.0),
        max_stale_s=cfg_float(config, "stream.max_stale_s", 2.0),
        stream_retry_interval_s=cfg_float(config, "stream.stream_retry_interval_s", 5.0),
        stream_fail_threshold=int(config.get("stream.stream_fail_threshold", 5)),
        stream_preflight=args.stream_preflight,
    )
    visualizer = DebugVisualizer(
        record_path=args.record_path,
        snapshot_dir=args.snapshot_dir,
    )
    estimator = TargetPoseEstimator(
        weights_path=str(weights_path),
        conf_threshold=conf_threshold,
        device=device,
    )
    controller = make_controller(config, servo_profile)

    start_time = time.monotonic()
    last_infer_time = 0.0
    last_report_time = start_time
    last_output_time = 0.0
    last_snapshot_time = start_time
    last_update_count = streamer.frame_update_count
    interval_reads = 0
    total_reads = 0
    total_none = 0
    source_fps = 0.0
    read_fps = 0.0
    final_mode = "closed"
    latest_pose: dict | None = None
    latest_debug: dict | None = None
    latest_velocity = (0.0, 0.0, 0.0)
    latest_detection_time = 0.0

    infer_interval_s = 1.0 / args.infer_hz if args.infer_hz > 0 else 0.0
    output_interval_s = 1.0 / args.output_hz if args.output_hz > 0 else 0.0

    try:
        while True:
            now = time.monotonic()
            if args.duration > 0 and now - start_time >= args.duration:
                break

            frame = streamer.get_latest_frame()
            if frame is None:
                total_none += 1
                time.sleep(0.01)
                continue

            total_reads += 1
            interval_reads += 1

            if output_interval_s > 0 and now - last_output_time < output_interval_s:
                time.sleep(0.001)
                continue
            last_output_time = now

            annotated = frame.copy()
            h, w = annotated.shape[:2]

            if args.use_config_center:
                center_u = args.center_u if args.center_u is not None else cfg_float(config, "camera.center_u", w / 2)
                center_v = args.center_v if args.center_v is not None else cfg_float(config, "camera.center_v", h / 2)
            else:
                center_u = args.center_u if args.center_u is not None else w / 2
                center_v = args.center_v if args.center_v is not None else h / 2
            center = (int(round(center_u)), int(round(center_v)))

            should_infer = infer_interval_s == 0.0 or now - last_infer_time >= infer_interval_s
            if should_infer:
                infer_dt = now - last_infer_time if last_infer_time > 0 else 0.0
                last_infer_time = now
                latest_pose = estimator.process_frame(frame, target_cls_id=target_cls_id)
                if latest_pose is not None:
                    latest_debug = controller.compute_debug(
                        latest_pose,
                        center_u=center_u,
                        center_v=center_v,
                        dt=infer_dt,
                    )
                    latest_velocity = latest_debug["velocities"]
                    latest_detection_time = now
                else:
                    latest_debug = None
                    latest_velocity = (0.0, 0.0, 0.0)
                    controller.reset()

            pose_is_fresh = (
                latest_pose is not None
                and now - latest_detection_time <= args.detection_stale_s
            )
            if pose_is_fresh:
                color = CLASS_COLORS.get(args.target, CLASS_COLORS["default"])
                visualizer.draw_obb(
                    annotated,
                    u=latest_pose["u"],
                    v=latest_pose["v"],
                    w=latest_pose["w"],
                    h=latest_pose["h"],
                    theta=latest_pose["theta"],
                    label=args.target,
                    conf=latest_pose["conf"],
                    color=color,
                )
                visualizer.draw_error_vector(
                    annotated,
                    center=center,
                    target=(int(round(latest_pose["u"])), int(round(latest_pose["v"]))),
                )
            else:
                cv2.drawMarker(annotated, center, (0, 0, 255), cv2.MARKER_CROSS, 16, 1)

            report_dt = now - last_report_time
            if report_dt >= args.report_interval_s:
                update_count = streamer.frame_update_count
                source_fps = (update_count - last_update_count) / report_dt
                read_fps = interval_reads / report_dt
                last_update_count = update_count
                interval_reads = 0
                last_report_time = now
                print(
                    f"mode={streamer.current_mode:>7} "
                    f"source_fps={source_fps:5.1f} read/s={read_fps:6.1f} "
                    f"target={'yes' if pose_is_fresh else 'no'}"
                )

            vx, vy, vyaw = latest_velocity
            hud = {
                "state": "VISUAL_HUD",
                "camera": "esp32_s3",
                "mode": streamer.current_mode,
                "target": args.target if pose_is_fresh else "none",
                "source_fps": source_fps,
                "read_fps": read_fps,
                "fps": source_fps,
                "dt": latest_debug["dt"] if latest_debug is not None else 0.0,
                "vx": vx,
                "vy": vy,
                "vyaw": vyaw,
            }
            if pose_is_fresh:
                hud["conf"] = latest_pose["conf"]
            if latest_debug is not None and pose_is_fresh:
                err_x, err_y, err_yaw = latest_debug["errors"]
                p_x, p_y, p_yaw = latest_debug["p_terms"]
                d_x, d_y, d_yaw = latest_debug["d_terms"]
                hud.update(
                    {
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

            visualizer.draw_hud(annotated, hud)
            visualizer.write_frame(annotated)

            if args.snapshot_every_s > 0 and now - last_snapshot_time >= args.snapshot_every_s:
                visualizer.save_frame(annotated, tag=args.target)
                last_snapshot_time = now

            if display_enabled:
                cv2.imshow(args.window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        final_mode = streamer.current_mode
        streamer.release()
        visualizer.release()
        if display_enabled:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - start_time
    print()
    print("=== Visual HUD summary ===")
    print(f"  elapsed       : {elapsed:.2f}s")
    print(f"  valid reads   : {total_reads}")
    print(f"  none frames   : {total_none}")
    print(f"  final mode    : {final_mode}")
    print(f"  final source/s: {source_fps:.1f}")


if __name__ == "__main__":
    main()
