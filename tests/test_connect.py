import argparse
import logging
import sys
import time
from pathlib import Path

# 允许直接运行 python3 tests/test_connect.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig


def parse_args():
    parser = argparse.ArgumentParser(description="FlightBridge 分层调试脚本")
    parser.add_argument("--conn", default="/dev/ttyUSB0", help="连接串，如 /dev/ttyUSB0 或 tcp:127.0.0.1:5760")
    parser.add_argument("--baud", type=int, default=57600, help="串口波特率")
    parser.add_argument("--timeout", type=int, default=20, help="心跳超时秒数")
    parser.add_argument("--samples", type=int, default=8, help="遥测采样次数")
    parser.add_argument("--interval", type=float, default=1.0, help="采样间隔秒")
    parser.add_argument("--mode", default="", help="可选：测试模式切换，如 GUIDED/LOITER/RTL")
    return parser.parse_args()


def attach_status_listener(vehicle, status_cache):
    # 捕获飞控状态文本，便于定位 PreArm 或 failsafe
    @vehicle.on_message("STATUSTEXT")
    def _on_status(self, name, msg):
        txt = getattr(msg, "text", "")
        sev = getattr(msg, "severity", -1)
        line = "STATUSTEXT severity={} text={}".format(sev, txt)
        status_cache.append(line)
        logging.warning(line)


def print_basic_info(vehicle):
    fw = getattr(vehicle, "version", None)
    print("firmware_version:", fw)
    print("mode:", vehicle.mode.name if vehicle.mode else "UNKNOWN")
    print("armed:", vehicle.armed)
    print("heading:", vehicle.heading)
    try:
        frame_class = vehicle.parameters.get("FRAME_CLASS", None)
        frame_type = vehicle.parameters.get("FRAME_TYPE", None)
        print("FRAME_CLASS:", frame_class)
        print("FRAME_TYPE:", frame_type)
    except Exception as ex:
        print("read FRAME params failed:", ex)


def telemetry_probe(fb, samples, interval):
    print("==== telemetry probe start ====")
    for i in range(samples):
        tel = fb.get_telemetry()
        print(
            "[{}] heartbeat_ok={} mode={} armed={} alt={:.2f} heading={} battery_pct={:.2f}".format(
                i + 1,
                tel.get("heartbeat_ok"),
                tel.get("mode"),
                tel.get("armed"),
                float(tel.get("alt", 0.0) or 0.0),
                tel.get("heading"),
                float(tel.get("battery_pct", 0.0) or 0.0),
            )
        )
        time.sleep(interval)
    print("==== telemetry probe end ====")


def mode_probe(fb, target_mode):
    if not target_mode:
        return
    print("==== mode probe start ====")
    ok = fb.set_mode(target_mode)
    print("set_mode({}) => {}".format(target_mode, ok))
    print("current_mode:", fb.get_telemetry().get("mode"))
    print("==== mode probe end ====")


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    cfg = FlightConfig(
        connection_string=args.conn,
        heartbeat_timeout=args.timeout,
        pixhawk_baud=args.baud,
        takeoff_timeout_s=45,
        land_timeout_s=90,
        land_detect_alt=0.15,
    )

    fb = FlightBridge(cfg)
    status_cache = []

    print("try connect:", args.conn, "baud=", args.baud, "timeout=", args.timeout)
    ok = fb.connect()
    print("connect:", ok)
    if not ok:
        print("连接失败：请先检查串口占用、波特率、数传参数、SERIALx_PROTOCOL/SERIALx_BAUD")
        return

    try:
        vehicle = fb._vehicle
        attach_status_listener(vehicle, status_cache)
        print_basic_info(vehicle)
        telemetry_probe(fb, args.samples, args.interval)
        mode_probe(fb, args.mode)

        if status_cache:
            print("==== captured STATUSTEXT (last 10) ====")
            for line in status_cache[-10:]:
                print(line)
    finally:
        if fb._vehicle is not None:
            fb._vehicle.close()
            print("vehicle closed")


if __name__ == "__main__":
    main()