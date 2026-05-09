#!/usr/bin/env python3
"""
无人机实时状态监测脚本
用于实时观测飞控状态、电池、高度、模式等信息

功能：
1. 实时显示飞控连接状态
2. 监测飞行模式、高度、电池电量
3. 心跳监控
4. 可选的 CSV 日志记录
5. 友好的终端界面
"""

import argparse
import logging
import signal
import sys
import time
import csv
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig

# 全局标志
should_exit = False


def signal_handler(sig, frame):
    """Ctrl+C 信号处理器"""
    global should_exit
    print("\n[INFO] 检测到 Ctrl+C，正在退出...")
    should_exit = True


def setup_logger(log_file: str = None):
    """配置日志"""
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
    return logging.getLogger("DroneMonitor")


def format_telemetry(tel: dict) -> str:
    """格式化遥测数据为友好的字符串"""
    status = "✓" if tel.get("heartbeat_ok", False) else "✗"
    armed_status = "🔓 已解锁" if tel.get("armed", False) else "🔒 未解锁"
    
    output = f"""
╔════════════════════════════════════════════════════════╗
║           无人机实时状态监测 - {datetime.now().strftime('%H:%M:%S')}           ║
╚════════════════════════════════════════════════════════╝

📡 连接状态：{status} {'正常' if tel.get('heartbeat_ok', False) else '异常'}
🔐 解锁状态：{armed_status}
✈️  飞行模式：{tel.get('mode', 'N/A')}

📊 飞行参数：
   ├─ 高度：{tel.get('alt', 0):.2f} m
   ├─ 航向：{tel.get('heading', 0):.1f}°
   └─ 电池：{tel.get('battery_pct', 0)*100:.1f}%

"""
    return output


def create_csv_writer(csv_file: str):
    """创建 CSV 日志写入器"""
    path = Path(csv_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    file = open(csv_file, 'w', newline='')
    writer = csv.DictWriter(
        file,
        fieldnames=['timestamp', 'armed', 'mode', 'alt', 'heading', 'battery_pct', 'heartbeat_ok']
    )
    writer.writeheader()
    file.flush()
    
    return file, writer


def monitor_drone(
    fb: FlightBridge,
    logger,
    update_interval_s: float = 1.0,
    csv_writer = None,
    csv_file = None
):
    """
    监测无人机状态的主循环
    
    Args:
        fb: FlightBridge 实例
        logger: 日志记录器
        update_interval_s: 更新间隔（秒）
        csv_writer: CSV 写入器（可选）
        csv_file: CSV 文件对象（可选）
    """
    global should_exit
    
    logger.info("=" * 60)
    logger.info("开始监测无人机状态...")
    logger.info("=" * 60)
    
    if csv_writer:
        logger.info(f"日志将保存到: {csv_file.name}")
    
    logger.info("按 Ctrl+C 退出监测\n")
    
    last_print_time = 0
    update_count = 0
    
    while not should_exit:
        try:
            current_time = time.time()
            
            # 获取遥测数据
            tel = fb.get_telemetry()
            
            # 定期输出到终端
            if current_time - last_print_time >= update_interval_s:
                # 清屏（可选）
                # os.system('clear' if os.name == 'posix' else 'cls')
                print(format_telemetry(tel), end='')
                last_print_time = current_time
                update_count += 1
            
            # 写入 CSV 日志
            if csv_writer:
                csv_writer.writerow({
                    'timestamp': datetime.now().isoformat(),
                    'armed': tel.get('armed', False),
                    'mode': tel.get('mode', ''),
                    'alt': tel.get('alt', 0.0),
                    'heading': tel.get('heading', 0.0),
                    'battery_pct': tel.get('battery_pct', 0.0),
                    'heartbeat_ok': tel.get('heartbeat_ok', False)
                })
                csv_file.flush()
            
            time.sleep(0.1)  # 降低 CPU 占用
            
        except KeyboardInterrupt:
            logger.info("检测到中断信号")
            should_exit = True
            break
        except Exception as e:
            logger.error(f"监测过程中出错: {e}", exc_info=True)
            time.sleep(1)
    
    logger.info("=" * 60)
    logger.info(f"监测结束，共采集 {update_count} 次状态快照")
    logger.info("=" * 60)


def main():
    global should_exit
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 解析参数
    parser = argparse.ArgumentParser(
        description="无人机实时状态监测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基本连接（默认 UDP 仿真器）
  python3 monitor_drone.py
  
  # 连接到串口飞控
  python3 monitor_drone.py --conn /dev/ttyUSB0 --baud 57600
  
  # 连接并保存日志
  python3 monitor_drone.py --conn /dev/ttyUSB0 --log-csv logs/flight_monitor.csv
  
  # 自定义更新频率（0.5 秒）
  python3 monitor_drone.py --interval 0.5
        """
    )
    
    parser.add_argument(
        "--conn",
        default="tcp:127.0.0.1:5760",
        help="飞控连接串，如 /dev/ttyUSB0 或 tcp:127.0.0.1:5760 (默认: %(default)s)"
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=57600,
        help="波特率 (默认: %(default)s)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="状态更新间隔，秒 (默认: %(default)s)"
    )
    parser.add_argument(
        "--log-csv",
        help="CSV 日志文件路径（如果指定，将记录所有状态）"
    )
    parser.add_argument(
        "--log-file",
        help="文本日志文件路径"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger(args.log_file)
    
    # 创建配置和飞控实例
    config = FlightConfig(
        connection_string=args.conn,
        pixhawk_baud=args.baud,
        heartbeat_timeout=20,
    )
    
    fb = FlightBridge(config)
    
    logger.info("=" * 60)
    logger.info("无人机状态监测脚本")
    logger.info("=" * 60)
    logger.info(f"连接串: {args.conn}")
    logger.info(f"波特率: {args.baud}")
    logger.info(f"更新间隔: {args.interval}s")
    
    # 连接飞控
    logger.info("\n正在连接飞控...")
    if not fb.connect():
        logger.error("❌ 连接失败！")
        logger.error("请检查：")
        logger.error("  1. 飞控电源是否接通")
        logger.error("  2. 连接串/波特率是否正确")
        logger.error("  3. 数据线是否正确连接")
        return 1
    
    logger.info("✓ 连接成功！")
    
    # 获取初始状态
    tel = fb.get_telemetry()
    logger.info(f"\n初始状态：")
    logger.info(f"  模式: {tel.get('mode', 'N/A')}")
    logger.info(f"  解锁: {tel.get('armed', False)}")
    logger.info(f"  高度: {tel.get('alt', 0):.2f}m")
    logger.info(f"  电池: {tel.get('battery_pct', 0)*100:.1f}%")
    
    # 创建 CSV 日志写入器（如果指定）
    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_file, csv_writer = create_csv_writer(args.log_csv)
        logger.info(f"✓ CSV 日志已创建: {args.log_csv}")
    
    try:
        # 开始监测
        monitor_drone(
            fb,
            logger,
            update_interval_s=args.interval,
            csv_writer=csv_writer,
            csv_file=csv_file
        )
        
        return 0
        
    except Exception as e:
        logger.critical(f"监测过程发生异常: {e}", exc_info=True)
        return 1
        
    finally:
        # 清理资源
        if csv_file:
            csv_file.close()
            logger.info(f"✓ CSV 日志已关闭")
        
        if fb._vehicle is not None:
            logger.info("正在关闭连接...")
            fb._vehicle.close()
            logger.info("✓ 连接已关闭")


if __name__ == "__main__":
    sys.exit(main())