"""
Live integration test for M1 ZeroLatencyStreamer.

This script intentionally probes the camera with Python requests before starting
the streamer. On WSL2, Windows browsers/curl may reach the ESP32 camera while
Linux Python sockets cannot route to the same 192.168.x subnet. The diagnostic
output helps separate streamer bugs from host networking problems.

Examples:
    python scripts/test_streamer_live.py --diagnose
    python scripts/test_streamer_live.py --prefer-capture --diagnose
    python scripts/test_streamer_live.py \
        --stream-url http://172.23.32.1:18081/stream \
        --capture-url http://172.23.32.1:18080/capture \
        --diagnose
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import cv2
import numpy as np
import requests

sys.path.insert(0, ".")
from core.streamer import ZeroLatencyStreamer


def _derived_stream_url(host: str) -> str:
    host = host.strip().rstrip("/")
    if host.startswith("http://") or host.startswith("https://"):
        base = host
    else:
        base = f"http://{host}"
    scheme, rest = base.split("://", 1)
    netloc = rest.split("/", 1)[0]
    if ":" not in netloc:
        netloc = f"{netloc}:81"
    return f"{scheme}://{netloc}/stream"


def _derived_capture_url(host: str) -> str:
    host = host.strip().rstrip("/")
    if host.startswith("http://") or host.startswith("https://"):
        base = host
    else:
        base = f"http://{host}"
    return f"{base}/capture"


def build_urls(args: argparse.Namespace) -> tuple[str, str]:
    stream_url = args.stream_url or _derived_stream_url(args.host)
    capture_url = args.capture_url or _derived_capture_url(args.host)
    return stream_url, capture_url


def _run_command(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<unavailable: {exc}>"

    output = (completed.stdout or completed.stderr).strip()
    return output or "<no output>"


def diagnose_network(host: str, stream_url: str, capture_url: str, timeout_s: float) -> None:
    capture_ok = False
    stream_ok = False

    print("=== Python/WSL network diagnostics ===")
    print(f"ip route get {host}:")
    print(f"  {_run_command(['ip', 'route', 'get', host])}")
    print("ip route:")
    for line in _run_command(["ip", "route"]).splitlines():
        print(f"  {line}")

    print()
    print("[probe] GET /capture with Python requests")
    try:
        resp = requests.get(capture_url, timeout=timeout_s)
        print(f"  status       : {resp.status_code}")
        print(f"  content-type : {resp.headers.get('content-type', '<missing>')}")
        print(f"  bytes        : {len(resp.content)}")
        resp.raise_for_status()

        arr = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("  decode       : FAIL (cv2.imdecode returned None)")
        else:
            h, w, c = frame.shape
            print(f"  decode       : OK ({w}x{h}, channels={c}, dtype={frame.dtype})")
            capture_ok = True
    except requests.exceptions.RequestException as exc:
        print(f"  result       : FAIL ({type(exc).__name__}: {exc})")

    print()
    print("[probe] GET /stream headers with Python requests")
    try:
        resp = requests.get(stream_url, stream=True, timeout=timeout_s)
        print(f"  status       : {resp.status_code}")
        print(f"  content-type : {resp.headers.get('content-type', '<missing>')}")
        try:
            resp.raise_for_status()
        finally:
            resp.close()
        print("  result       : OK")
        stream_ok = True
    except requests.exceptions.RequestException as exc:
        print(f"  result       : FAIL ({type(exc).__name__}: {exc})")

    if (
        host.startswith("192.168.")
        and "172." in _run_command(["ip", "route", "get", host])
        and not (capture_ok and stream_ok)
    ):
        print()
        print("WSL2 hint:")
        print("  Python is using the Linux network stack. If Windows browser works but")
        print("  these probes fail, run this test from Windows Python, or expose a")
        print("  Windows-side HTTP/port proxy and pass its URLs with --stream-url and")
        print("  --capture-url.")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1 ZeroLatencyStreamer live test")
    parser.add_argument("--host", default="192.168.43.192")
    parser.add_argument("--stream-url", default=None, help="Override MJPEG stream URL")
    parser.add_argument("--capture-url", default=None, help="Override JPEG capture URL")
    parser.add_argument("--duration", type=float, default=8.0, help="Test duration in seconds")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--wait-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-stale-s", type=float, default=2.0)
    parser.add_argument("--capture-poll-interval-s", type=float, default=1.0 / 15.0)
    parser.add_argument("--stream-retry-interval-s", type=float, default=5.0)
    parser.add_argument("--stream-fail-threshold", type=int, default=5)
    parser.add_argument("--prefer-capture", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--no-stream-preflight", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stream_url, capture_url = build_urls(args)
    prefer_stream = not args.prefer_capture and not args.no_stream

    print("=== M1 ZeroLatencyStreamer live test ===")
    print(f"  host                  : {args.host}")
    print(f"  stream_url            : {stream_url}")
    print(f"  capture_url           : {capture_url}")
    print(f"  prefer_stream         : {prefer_stream}")
    print(f"  stream_preflight      : {not args.no_stream_preflight}")
    print(f"  timeout_ms            : {args.timeout_ms}")
    print(f"  wait_timeout_s        : {args.wait_timeout_s}")
    print(f"  duration_s            : {args.duration}")
    print()

    if args.diagnose:
        diagnose_network(args.host, stream_url, capture_url, args.timeout_ms / 1000.0)

    streamer = ZeroLatencyStreamer(
        host=args.host,
        stream_url=stream_url,
        capture_url=capture_url,
        timeout_ms=args.timeout_ms,
        prefer_stream=prefer_stream,
        capture_poll_interval_s=args.capture_poll_interval_s,
        max_stale_s=args.max_stale_s,
        stream_fail_threshold=args.stream_fail_threshold,
        stream_retry_interval_s=args.stream_retry_interval_s,
        stream_preflight=not args.no_stream_preflight,
    )

    print("Waiting for first frame...")
    t_start_wait = time.monotonic()
    first_frame = None
    while time.monotonic() - t_start_wait < args.wait_timeout_s:
        first_frame = streamer.get_latest_frame()
        if first_frame is not None:
            break
        time.sleep(0.05)

    if first_frame is None:
        print(f"FAIL: no frame received within {args.wait_timeout_s:.1f}s.")
        print(f"mode     : {streamer.current_mode}")
        print(f"is_opened: {streamer.is_opened()}")
        streamer.release()
        sys.exit(1)

    wait_elapsed = time.monotonic() - t_start_wait
    print(f"First frame arrived after {wait_elapsed:.2f}s")
    print(f"  resolution: {first_frame.shape[1]}x{first_frame.shape[0]}")
    print(f"  dtype     : {first_frame.dtype}")
    print(f"  channels  : {first_frame.shape[2]}")
    print(f"  mode      : {streamer.current_mode}")
    print()

    print(f"Starting continuous source-rate test ({args.duration:.1f}s)...")
    print(
        "Note: source/s is counted when the background streamer writes a new "
        "frame into the cache; read/s is only the consumer polling rate."
    )
    print(
        f"{'time(s)':>8} | {'source/s':>8} | {'read/s':>6} | "
        f"{'resolution':>12} | {'mode':>8} | {'valid':>6} | {'none':>6}"
    )
    print("-" * 82)

    t_test_start = time.monotonic()
    start_update_count = streamer.frame_update_count
    last_update_count = start_update_count
    total_valid = 0
    total_none = 0
    interval_valid = 0
    interval_none = 0
    last_report = t_test_start
    resolutions_seen: set[str] = set()
    modes_seen: set[str] = set()

    while True:
        now = time.monotonic()
        elapsed = now - t_test_start
        if elapsed >= args.duration:
            break

        frame = streamer.get_latest_frame()
        if frame is not None:
            total_valid += 1
            interval_valid += 1
            h, w, _ = frame.shape
            resolutions_seen.add(f"{w}x{h}")
        else:
            total_none += 1
            interval_none += 1

        mode = streamer.current_mode
        modes_seen.add(mode)

        if now - last_report >= 1.0:
            dt = now - last_report
            update_count = streamer.frame_update_count
            interval_updates = update_count - last_update_count
            source_rate = interval_updates / dt if dt > 0 else 0.0
            read_rate = interval_valid / dt if dt > 0 else 0.0
            resolution_text = "|".join(sorted(resolutions_seen)) or "-"
            print(
                f"{elapsed:8.1f} | {source_rate:8.1f} | {read_rate:6.1f} | "
                f"{resolution_text:>12} | {mode:>8} | "
                f"{interval_valid:6d} | {interval_none:6d}"
            )
            interval_valid = 0
            interval_none = 0
            last_update_count = update_count
            last_report = now

        time.sleep(0.001)

    total_elapsed = time.monotonic() - t_test_start
    source_updates = streamer.frame_update_count - start_update_count
    avg_source_rate = source_updates / total_elapsed if total_elapsed > 0 else 0.0
    avg_read_rate = total_valid / total_elapsed if total_elapsed > 0 else 0.0

    print()
    print("=== Final stats ===")
    print(f"  duration       : {total_elapsed:.2f}s")
    print(f"  source updates : {source_updates}")
    print(f"  avg source/s   : {avg_source_rate:.1f}")
    print(f"  valid frames   : {total_valid}")
    print(f"  none frames    : {total_none}")
    print(f"  avg read/s     : {avg_read_rate:.1f}")
    print(f"  resolutions    : {resolutions_seen}")
    print(f"  modes          : {modes_seen}")
    print(f"  is_opened      : {streamer.is_opened()}")

    ok = True
    if total_valid == 0:
        print("  FAIL: no valid frame received")
        ok = False
    if not any("400" in r or "296" in r for r in resolutions_seen):
        print(f"  WARN: expected approximately 400x296, got {resolutions_seen}")
    else:
        print("  PASS: resolution matches expected camera size")

    if avg_source_rate < 1.0:
        print(f"  FAIL: average source/s={avg_source_rate:.1f} is too low")
        ok = False
    else:
        print(f"  PASS: average source/s={avg_source_rate:.1f}")

    if avg_read_rate < 1.0:
        print(f"  FAIL: average read/s={avg_read_rate:.1f} is too low")
        ok = False
    else:
        print(f"  PASS: average read/s={avg_read_rate:.1f}")

    streamer.release()
    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
