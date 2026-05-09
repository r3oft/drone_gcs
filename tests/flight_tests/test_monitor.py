import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig

config = FlightConfig(connection_string="/dev/ttyUSB0", pixhawk_baud=57600)
fb = FlightBridge(config)

if fb.connect():
    print("连接成功")
    
    # 强制油门到最低
    print("\n强制设置油门为最低...")
    fb._vehicle.channels.overrides['3'] = 1000
    time.sleep(1)
    
    # 切换到 STABILIZE 模式
    print("切换到 STABILIZE 模式...")
    fb._vehicle.mode = "STABILIZE"
    time.sleep(1)
    print(f"✓ 当前模式: {fb._vehicle.mode.name}\n")
    
    # 解锁
    if not fb._vehicle.armed:
        print("解锁电机...")
        fb._vehicle.armed = True
        time.sleep(2)
    
    if fb._vehicle.armed:
        print("✓ 电机已解锁！\n")
        print("=" * 60)
        print("大幅度油门变化测试（更容易看到转速变化）")
        print("=" * 60)
        
        # 用更大的跳跃
        test_values = [
            (1000, "零油门"),
            (1200, "低油门（20%）"),
            (1400, "中低油门（40%）"),
            (1600, "中油门（60%）"),
            (1800, "高油门（80%）"),
            (1000, "回到零油门")
        ]
        
        for throttle, desc in test_values:
            print(f"\n设置油门为 {throttle} µs ({desc})...")
            fb._vehicle.channels.overrides['3'] = throttle
            
            # 持续输出 2 秒，便于观察电机转速
            print("  发送中... 请观察电机转速")
            for i in range(10):
                time.sleep(0.2)
                try:
                    ch3 = fb._vehicle.channels['3']
                    print(f"    [{i+1}/10] CH3={ch3} µs", end="\r")
                except:
                    pass
            print()  # 换行
        
        # 上锁
        print("\n上锁电机...")
        fb._vehicle.armed = False
        time.sleep(1)
        print("✓ 电机已上锁")
    
    fb._vehicle.close()