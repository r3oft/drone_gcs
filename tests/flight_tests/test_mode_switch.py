import argparse
import logging
import sys
import time
from pathlib import Path

# 允许直接运行 python3 tests/test_mode_switch.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig


def parse_args():
    parser = argparse.ArgumentParser(description="FlightBridge 模式切换测试脚本")
    parser.add_argument("--conn", default="/dev/ttyUSB0", help="连接串，如 /dev/ttyUSB0 或 tcp:127.0.0.1:5760")
    parser.add_argument("--baud", type=int, default=57600, help="串口波特率")
    parser.add_argument("--timeout", type=int, default=20, help="心跳超时秒数")
    parser.add_argument("--target", default="GUIDED", help="目标模式，默认 GUIDED（支持 GUIDE 别名）")
    parser.add_argument("--wait", type=float, default=1.5, help="切换后等待秒数")
    return parser.parse_args()


def normalize_mode(mode: str) -> str:
    normalized = (mode or "").strip().upper()
    if normalized == "GUIDE":
        return "GUIDED"
    return normalized


def print_mode_snapshot(fb: FlightBridge, title: str):
    tel = fb.get_telemetry()
    print(
        "{}: heartbeat_ok={} mode={} armed={} alt={:.2f} heading={}".format(
            title,
            tel.get("heartbeat_ok"),
            tel.get("mode"),
            tel.get("armed"),
            float(tel.get("alt", 0.0) or 0.0),
            tel.get("heading"),
        )
    )


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    target_mode = normalize_mode(args.target)

    cfg = FlightConfig(
        connection_string=args.conn,
        heartbeat_timeout=args.timeout,
        pixhawk_baud=args.baud,
    )

    fb = FlightBridge(cfg)

    print("try connect:", args.conn, "baud=", args.baud, "timeout=", args.timeout)
    ok = fb.connect()
    print("connect:", ok)
    if not ok:
        print("连接失败：请先检查串口占用、波特率、数传参数、SERIALx_PROTOCOL/SERIALx_BAUD")
        return

    try:
        valid_modes = ["GUIDED_NOGPS", "LOITER", "RTL"]
        if target_mode not in valid_modes:
            print("目标模式无效：{}，仅支持 {}".format(target_mode, valid_modes))
            return

        print_mode_snapshot(fb, "before switch")

        print("==== mode switch start ====")
        switched = fb.set_mode(target_mode)
        print("set_mode({}) => {}".format(target_mode, switched))

        time.sleep(max(args.wait, 0.0))
        print_mode_snapshot(fb, "after switch")
        print("==== mode switch end ====")
    finally:
        if fb._vehicle is not None:
            fb._vehicle.close()
            print("vehicle closed")


if __name__ == "__main__":
    main()
