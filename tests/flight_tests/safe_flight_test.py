#!/usr/bin/env python3
"""
安全飞行测试脚本 - 带多重保护机制
用于首次起飞、悬停、降落测试

安全特性：
1. 高度硬限制（超过阈值强制降落）
2. 心跳监控（数传断连自动降落）
3. 电池电压监控（低电压自动返航）
4. 遥控器接管检测
5. 紧急停止信号处理
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig

# ============ 安全参数配置 ============
SAFETY_CONFIG = {
    "max_altitude_m": 2.5,           # 绝对高度上限（米）
    "min_battery_pct": 0.25,         # 最低电池百分比
    "heartbeat_check_interval": 1.0, # 心跳检查间隔（秒）
    "max_heartbeat_loss": 3.0,       # 最大心跳丢失时间（秒）
    "velocity_cmd_rate": 5.0,        # 速度指令发送频率（Hz）
}

# 全局标志：用于信号处理
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
    return logging.getLogger("SafeFlightTest")


def check_safety_conditions(fb: FlightBridge, logger) -> tuple[bool, str]:
    """
    检查安全条件

    Returns:
        (is_safe, reason) - 是否安全及原因
    """
    tel = fb.get_telemetry()

    # 1. 检查心跳
    if not tel.get("heartbeat_ok", False):
        return False, "心跳丢失"

    # 2. 检查高度
    current_alt = tel.get("alt", 0.0)
    if current_alt > SAFETY_CONFIG["max_altitude_m"]:
        return False, f"超过最大高度限制 ({current_alt:.2f}m > {SAFETY_CONFIG['max_altitude_m']}m)"

    # 3. 检查电池
    battery_pct = tel.get("battery_pct", 0.0)
    if battery_pct < SAFETY_CONFIG["min_battery_pct"]:
        return False, f"电池电量过低 ({battery_pct*100:.1f}% < {SAFETY_CONFIG['min_battery_pct']*100:.1f}%)"

    # 4. 检查模式（如果不是 GUIDED，说明遥控器接管了）
    mode = tel.get("mode", "")
    if mode != "GUIDED":
        return False, f"飞行模式已切换为 {mode}（可能是遥控器接管）"

    return True, "正常"


def emergency_land(fb: FlightBridge, logger):
    """紧急降落程序"""
    logger.critical("执行紧急降落...")

    # 尝试切换到 LAND 模式
    if fb.set_mode("LAND"):
        logger.info("已切换到 LAND 模式")
    else:
        logger.error("切换 LAND 模式失败，尝试发送零速度指令...")
        # 如果切换失败，发送零速度指令让无人机悬停
        for _ in range(10):
            fb.send_body_velocity(0, 0, 0, 0)
            time.sleep(0.2)


def safe_hover(fb: FlightBridge, duration_s: float, logger) -> bool:
    """
    安全悬停（带实时监控）

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
    cmd_interval = 1.0 / SAFETY_CONFIG["velocity_cmd_rate"]
    last_heartbeat_time = time.time()

    logger.info(f"开始安全悬停 {duration_s} 秒...")

    while time.time() - start_time < duration_s:
        # 检查紧急停止标志
        if emergency_stop_requested:
            logger.warning("检测到紧急停止请求")
            return False

        # 检查安全条件
        is_safe, reason = check_safety_conditions(fb, logger)
        if not is_safe:
            logger.error(f"安全检查失败: {reason}")
            return False

        # 更新心跳时间
        tel = fb.get_telemetry()
        if tel.get("heartbeat_ok", False):
            last_heartbeat_time = time.time()
        else:
            # 检查心跳丢失时长
            if time.time() - last_heartbeat_time > SAFETY_CONFIG["max_heartbeat_loss"]:
                logger.error("心跳丢失超时")
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

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)

    # 解析参数
    parser = argparse.ArgumentParser(description="安全飞行测试脚本")
    parser.add_argument("--conn", default="/dev/ttyUSB0", help="连接串，如 /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=57600, help="波特率")
    parser.add_argument("--alt", type=float, default=1.2, help="起飞高度（米）")
    parser.add_argument("--hover", type=float, default=5.0, help="悬停时长（秒）")
    parser.add_argument("--dry-run", action="store_true", help="仅连接测试，不起飞")
    args = parser.parse_args()

    logger = setup_logger()

    # 安全检查：起飞高度不能超过限制
    if args.alt > SAFETY_CONFIG["max_altitude_m"]:
        logger.error(f"起飞高度 {args.alt}m 超过安全限制 {SAFETY_CONFIG['max_altitude_m']}m")
        return 1

    # 创建配置
    config = FlightConfig(
        connection_string=args.conn,
        pixhawk_baud=args.baud,
        heartbeat_timeout=20,
        takeoff_timeout_s=30,
        land_timeout_s=60,
    )

    fb = FlightBridge(config)

    logger.info("=" * 60)
    logger.info("安全飞行测试脚本")
    logger.info("=" * 60)
    logger.info(f"连接串: {args.conn}")
    logger.info(f"波特率: {args.baud}")
    logger.info(f"起飞高度: {args.alt}m")
    logger.info(f"悬停时长: {args.hover}s")
    logger.info(f"最大高度限制: {SAFETY_CONFIG['max_altitude_m']}m")
    logger.info(f"最低电池: {SAFETY_CONFIG['min_battery_pct']*100:.1f}%")
    logger.info("=" * 60)
    logger.info("紧急操作提示:")
    logger.info("  1. 按 Ctrl+C 触发代码级紧急降落")
    logger.info("  2. 遥控器切换到 LAND/RTL 模式立即接管")
    logger.info("  3. 遥控器油门拉到最低并保持可强制上锁")
    logger.info("=" * 60)

    # 连接飞控
    logger.info("正在连接飞控...")
    if not fb.connect():
        logger.error("连接失败")
        return 1

    logger.info("连接成功")

    try:
        # 打印初始状态
        tel = fb.get_telemetry()
        logger.info(f"初始状态: 模式={tel.get('mode')} 解锁={tel.get('armed')} "
                   f"高度={tel.get('alt', 0):.2f}m 电池={tel.get('battery_pct', 0)*100:.1f}%")

        # 如果是 dry-run 模式，仅测试连接
        if args.dry_run:
            logger.info("Dry-run 模式，跳过起飞")
            return 0

        # 起飞前最后确认
        logger.warning("即将起飞，请确保:")
        logger.warning("  1. 螺旋桨已安装且紧固")
        logger.warning("  2. 周围环境安全，无人员靠近")
        logger.warning("  3. 遥控器已开机且信号正常")
        logger.warning("  4. 电池电量充足")
        logger.warning("按 Enter 继续，或 Ctrl+C 取消...")
        input()

        # 起飞
        logger.info(f"开始起飞到 {args.alt}m...")
        if not fb.arm_and_takeoff(args.alt):
            logger.error("起飞失败或超时")
            return 1

        logger.info(f"已到达目标高度 {args.alt}m")

        # 安全悬停
        if not safe_hover(fb, args.hover, logger):
            logger.error("悬停过程中检测到安全问题，执行紧急降落")
            emergency_land(fb, logger)
            return 1

        # 正常降落
        logger.info("准备降落...")
        if fb.land():
            logger.info("降落成功")
            return 0
        else:
            logger.error("降落超时")
            return 1

    except KeyboardInterrupt:
        logger.critical("检测到 Ctrl+C")
        emergency_land(fb, logger)
        return 1

    except Exception as e:
        logger.critical(f"发生异常: {e}", exc_info=True)
        emergency_land(fb, logger)
        return 1

    finally:
        if fb._vehicle is not None:
            logger.info("关闭连接...")
            fb._vehicle.close()


if __name__ == "__main__":
    sys.exit(main())