#!/usr/bin/env python3
"""
无遥控器安全飞行脚本 - 极度保守的安全策略

关键特性：
1. 更严格的高度限制（1.5m）
2. 更频繁的心跳检查（0.5s）
3. 自动降落的多个触发条件
4. 详细的日志记录
5. 飞控 Failsafe 作为最后防线

使用场景：
- 代码逻辑验证（拆桨）
- 基础飞行能力测试（系留）
- 传感器数据采集

⚠️ 警告：无遥控器意味着无法人工接管，仅在完全受控的环境中使用
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig

# ============ 无遥控器专用安全参数 ============
SAFETY_CONFIG_NO_RC = {
    "max_altitude_m": 1.5,              # 更低的高度限制
    "min_battery_pct": 0.35,            # 更高的电池阈值
    "heartbeat_check_interval": 0.5,    # 更频繁的检查
    "max_heartbeat_loss": 2.0,          # 更短的容忍时间
    "velocity_cmd_rate": 10.0,          # 更高的指令频率
    "emergency_land_timeout_s": 10.0,   # 紧急降落超时
}

emergency_stop_requested = False


def signal_handler(sig, frame):
    """Ctrl+C 信号处理器"""
    global emergency_stop_requested
    print("\n[紧急] 检测到 Ctrl+C，触发紧急降落程序...")
    emergency_stop_requested = True


def setup_logger():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("NoRCSafeFlightTest")


def check_safety_conditions_strict(fb: FlightBridge, logger) -> tuple[bool, str]:
    """
    严格的安全检查（无遥控器版本）

    Returns:
        (is_safe, reason)
    """
    tel = fb.get_telemetry()

    # 1. 心跳检查（最严格）
    if not tel.get("heartbeat_ok", False):
        return False, "心跳丢失 - 数传断连"

    # 2. 高度检查（更严格）
    current_alt = tel.get("alt", 0.0)
    if current_alt > SAFETY_CONFIG_NO_RC["max_altitude_m"]:
        return False, f"超过最大高度 ({current_alt:.2f}m > {SAFETY_CONFIG_NO_RC['max_altitude_m']}m)"

    # 3. 电池检查（更严格）
    battery_pct = tel.get("battery_pct", 0.0)
    if battery_pct < SAFETY_CONFIG_NO_RC["min_battery_pct"]:
        return False, f"电池过低 ({battery_pct*100:.1f}%)"

    # 4. 模式检查（无遥控器，模式不应被切换）
    mode = tel.get("mode", "")
    if mode != "GUIDED":
        return False, f"模式异常切换为 {mode}"

    # 5. 解锁状态检查
    if not tel.get("armed", False):
        return False, "飞控已上锁"

    return True, "正常"


def emergency_land_no_rc(fb: FlightBridge, logger) -> bool:
    """
    无遥控器紧急降落程序

    Returns:
        True: 成功降落
        False: 降落失败
    """
    logger.critical("执行无遥控器紧急降落...")

    start_time = time.time()
    timeout = SAFETY_CONFIG_NO_RC["emergency_land_timeout_s"]

    # 尝试切换到 LAND 模式
    if not fb.set_mode("LAND"):
        logger.error("切换 LAND 模式失败，尝试发送零速度指令...")
        # 发送零速度指令
        for i in range(50):  # 10 秒内持续发送
            fb.send_body_velocity(0, 0, 0, 0)
            time.sleep(0.2)

            tel = fb.get_telemetry()
            if not tel.get("armed", False):
                logger.info("飞控已上锁，降落成功")
                return True

        logger.error("紧急降落失败")
        return False

    logger.info("已切换到 LAND 模式，等待触地...")

    # 等待触地
    while time.time() - start_time < timeout:
        tel = fb.get_telemetry()

        # 检查是否已上锁
        if not tel.get("armed", False):
            logger.info("飞控已上锁，降落成功")
            return True

        # 检查高度
        current_alt = tel.get("alt", 0.0)
        if current_alt < 0.2:
            logger.info(f"已触地 (高度 {current_alt:.2f}m)")
            return True

        time.sleep(0.5)

    logger.error(f"紧急降落超时 ({timeout}s)")
    return False


def safe_hover_no_rc(fb: FlightBridge, duration_s: float, logger) -> bool:
    """
    无遥控器安全悬停（极度保守）

    Args:
        fb: FlightBridge 实例
        duration_s: 悬停时长（秒）
        logger: 日志记录器

    Returns:
        True: 悬停成功完成
        False: 因安全问题中断
    """
    global emergency_stop_requested

    start_time = time.time()
    cmd_interval = 1.0 / SAFETY_CONFIG_NO_RC["velocity_cmd_rate"]
    last_heartbeat_time = time.time()
    last_safety_check = time.time()

    logger.info(f"开始无遥控器安全悬停 {duration_s} 秒...")
    logger.warning("⚠️ 无遥控器模式 - 任何异常都会立即触发紧急降落")

    while time.time() - start_time < duration_s:
        # 检查紧急停止标志
        if emergency_stop_requested:
            logger.warning("检测到紧急停止请求")
            return False

        # 每 0.5 秒进行一次完整安全检查
        if time.time() - last_safety_check > 0.5:
            is_safe, reason = check_safety_conditions_strict(fb, logger)
            if not is_safe:
                logger.error(f"安全检查失败: {reason}")
                return False
            last_safety_check = time.time()

        # 获取遥测数据
        tel = fb.get_telemetry()

        # 更新心跳时间
        if tel.get("heartbeat_ok", False):
            last_heartbeat_time = time.time()
        else:
            # 检查心跳丢失时长
            if time.time() - last_heartbeat_time > SAFETY_CONFIG_NO_RC["max_heartbeat_loss"]:
                logger.error("心跳丢失超时 - 触发紧急降落")
                return False

        # 发送零速度指令保持悬停
        fb.send_body_velocity(0, 0, 0, 0)

        # 每秒打印一次状态
        elapsed = time.time() - start_time
        if int(elapsed) != int(elapsed - cmd_interval):
            logger.info(
                f"悬停中... {elapsed:.1f}s/{duration_s}s | "
                f"高度: {tel.get('alt', 0):.2f}m | "
                f"电池: {tel.get('battery_pct', 0)*100:.1f}% | "
                f"航向: {tel.get('heading', 0)}°"
            )

        time.sleep(cmd_interval)

    logger.info("悬停完成")
    return True


def main():
    global emergency_stop_requested

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="无遥控器安全飞行脚本")
    parser.add_argument("--conn", default="/dev/ttyUSB0", help="连接串")
    parser.add_argument("--baud", type=int, default=57600, help="波特率")
    parser.add_argument("--alt", type=float, default=0.8, help="起飞高度（米）")
    parser.add_argument("--hover", type=float, default=3.0, help="悬停时长（秒）")
    parser.add_argument("--dry-run", action="store_true", help="仅连接测试")
    args = parser.parse_args()

    logger = setup_logger()

    # 安全检查
    if args.alt > SAFETY_CONFIG_NO_RC["max_altitude_m"]:
        logger.error(f"起飞高度 {args.alt}m 超过无遥控器安全限制 {SAFETY_CONFIG_NO_RC['max_altitude_m']}m")
        return 1

    config = FlightConfig(
        connection_string=args.conn,
        pixhawk_baud=args.baud,
        heartbeat_timeout=10,  # 更短的超时
        takeoff_timeout_s=20,
        land_timeout_s=30,
    )

    fb = FlightBridge(config)

    logger.info("=" * 70)
    logger.info("🔴 无遥控器安全飞行脚本 - 极度保守模式")
    logger.info("=" * 70)
    logger.info(f"连接串: {args.conn}")
    logger.info(f"波特率: {args.baud}")
    logger.info(f"起飞高度: {args.alt}m (限制: {SAFETY_CONFIG_NO_RC['max_altitude_m']}m)")
    logger.info(f"悬停时长: {args.hover}s")
    logger.info(f"最低电池: {SAFETY_CONFIG_NO_RC['min_battery_pct']*100:.1f}%")
    logger.info(f"心跳超时: {SAFETY_CONFIG_NO_RC['max_heartbeat_loss']}s")
    logger.info("=" * 70)
    logger.warning("⚠️ 警告：无遥控器意味着无法人工接管")
    logger.warning("⚠️ 仅在完全受控的环境中使用此脚本")
    logger.warning("⚠️ 任何异常都会立即触发紧急降落")
    logger.warning("⚠️ 按 Ctrl+C 可随时触发紧急降落")
    logger.info("=" * 70)

    # 连接飞控
    logger.info("正在连接飞控...")
    if not fb.connect():
        logger.error("连接失败")
        return 1

    logger.info("连接成功")

    try:
        tel = fb.get_telemetry()
        logger.info(f"初始状态: 模式={tel.get('mode')} 解锁={tel.get('armed')} "
                   f"高度={tel.get('alt', 0):.2f}m 电池={tel.get('battery_pct', 0)*100:.1f}%")

        if args.dry_run:
            logger.info("Dry-run 模式，跳过起飞")
            return 0

        # 最后确认
        logger.warning("即将起飞（无遥控器模式）")
        logger.warning("确保：")
        logger.warning("  1. 飞控 Failsafe 已启用（FS_GCS_ENABLE=1）")
        logger.warning("  2. 周围环境完全安全")
        logger.warning("  3. 系留绳索已安装")
        logger.warning("按 Enter 继续，或 Ctrl+C 取消...")
        input()

        # 起飞
        logger.info(f"开始起飞到 {args.alt}m...")
        if not fb.arm_and_takeoff(args.alt):
            logger.error("起飞失败")
            return 1

        logger.info(f"已到达目标高度 {args.alt}m")

        # 安全悬停
        if not safe_hover_no_rc(fb, args.hover, logger):
            logger.error("悬停过程中检测到安全问题")
            emergency_land_no_rc(fb, logger)
            return 1

        # 降落
        logger.info("准备降落...")
        if fb.land():
            logger.info("降落成功")
            return 0
        else:
            logger.error("降落超时")
            return 1

    except KeyboardInterrupt:
        logger.critical("检测到 Ctrl+C")
        emergency_land_no_rc(fb, logger)
        return 1

    except Exception as e:
        logger.critical(f"发生异常: {e}", exc_info=True)
        emergency_land_no_rc(fb, logger)
        return 1

    finally:
        if fb._vehicle is not None:
            logger.info("关闭连接...")
            fb._vehicle.close()


if __name__ == "__main__":
    sys.exit(main())
