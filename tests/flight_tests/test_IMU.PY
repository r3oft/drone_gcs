#!/usr/bin/env python3
import sys
print("1. Python 启动成功", flush=True)

from pathlib import Path
print("2. pathlib 导入成功", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(f"3. PROJECT_ROOT: {PROJECT_ROOT}", flush=True)

sys.path.insert(0, str(PROJECT_ROOT))
print("4. sys.path 已更新", flush=True)

try:
    from core.flight_bridge import FlightBridge, FlightConfig
    print("5. FlightBridge 导入成功", flush=True)
except Exception as e:
    print(f"❌ FlightBridge 导入失败: {e}", flush=True)
    sys.exit(1)

try:
    from dronekit import VehicleMode
    print("6. dronekit 导入成功", flush=True)
except Exception as e:
    print(f"❌ dronekit 导入失败: {e}", flush=True)
    sys.exit(1)

import time
print("7. time 导入成功", flush=True)

print("\n开始连接飞控...", flush=True)
try:
    config = FlightConfig(connection_string="/dev/ttyUSB0", pixhawk_baud=57600)
    print("8. FlightConfig 创建成功", flush=True)
    
    fb = FlightBridge(config)
    print("9. FlightBridge 创建成功", flush=True)
    
    print("10. 正在连接...", flush=True)
    if fb.connect():
        print("11. ✅ 连接成功!", flush=True)
        vehicle = fb._vehicle
        print(f"12. 飞控模式: {vehicle.mode.name}", flush=True)
        vehicle.close()
    else:
        print("11. ❌ 连接失败", flush=True)
        
except Exception as e:
    print(f"❌ 异常: {e}", flush=True)
    import traceback
    traceback.print_exc()