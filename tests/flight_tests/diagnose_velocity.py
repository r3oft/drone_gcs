import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig

print("=" * 70)
print("SET_ATTITUDE_TARGET vs SET_POSITION_TARGET_LOCAL_NED 对比测试")
print("=" * 70)

config = FlightConfig(connection_string="/dev/ttyUSB0", pixhawk_baud=57600)
fb = FlightBridge(config)

if fb.connect():
    vehicle = fb._vehicle
    
    # 【准备环境】
    print("\n【准备测试环境】")
    print("-" * 70)
    
    print("  强制油门最低...")
    vehicle.channels.overrides['3'] = 1000
    time.sleep(1)
    
    print("  切换到 GUIDED_NOGPS...")
    vehicle.mode = "GUIDED_NOGPS"
    time.sleep(1)
    print(f"  当前模式: {vehicle.mode.name}")
    
    print("  解锁电机...")
    vehicle.armed = True
    time.sleep(2)
    
    if not vehicle.armed:
        print("  ❌ 解锁失败")
        vehicle.close()
        sys.exit(1)
    
    print("  ✓ 准备完成，等待 2 秒...")
    time.sleep(2)
    
    # # 【测试 A】SET_POSITION_TARGET_LOCAL_NED（速度指令）
    # print("\n【测试 A】SET_POSITION_TARGET_LOCAL_NED（速度指令）")
    # print("-" * 70)
    # print("  发送向上速度指令 10 次：vz = -0.5 m/s")
    
    # ch_before = vehicle.channels['3']
    # print(f"  初始 PWM: {ch_before}")
    
    # for i in range(10):
    #     fb.send_body_velocity(0, 0, -0.5, 0)
    #     time.sleep(0.2)
    
    # ch_after = vehicle.channels['3']
    # print(f"  最终 PWM: {ch_after}")
    # result_a = "✅ 有效" if ch_after > ch_before else "❌ 无效"
    # print(f"  结果: {result_a}")
    
    # # 重置
    # vehicle.channels.overrides['3'] = 1520
    # time.sleep(1)
    
    # 【测试 B】SET_ATTITUDE_TARGET（姿态指令）
    print("\n【测试 B】SET_ATTITUDE_TARGET（姿态指令）")
    print("-" * 70)
    print("  发送上升油门指令 10 次：thrust = 0.6（悬停 ~0.5）")
    
    ch_before = vehicle.channels['3']
    print(f"  初始 PWM: {ch_before}")
    
    for i in range(10):
        fb.send_attitude_target(0, 0, 0, 0.7)  # 静止姿态 + 60% 油门
        time.sleep(2)
        fb.send_attitude_target(0, 0, 0, 0)
    
    ch_after = vehicle.channels['3']
    print(f"  最终 PWM: {ch_after}")
    result_b = "✅ 有效" if ch_after > ch_before else "❌ 无效"
    print(f"  结果: {result_b}")
    
    # 重置
    vehicle.channels.overrides['3'] = 0
    time.sleep(1)
    
    # 【诊断结论】
    print("\n【诊断结论】")
    print("-" * 70)
    print(f"""
  固件版本: ArduCopter V3.6.7
  测试模式: GUIDED_NOGPS
  
  SET_POSITION_TARGET_LOCAL_NED 结果: {result_a}
  SET_ATTITUDE_TARGET 结果:           {result_b}
  
  分析：
  - 如果两个都无效 → 固件 3.6.7 对 MAVLink 速度/姿态控制支持有限
  - 如果只有 SET_ATTITUDE_TARGET 有效 → 使用此方法代替
  - 如果都有效 → 优先使用 SET_ATTITUDE_TARGET（更直接）
    """)
    
    # 清理
    print("\n【清理】")
    print("-" * 70)
    # vehicle.channels.overrides['3'] = 1000
    time.sleep(1)
    vehicle.armed = False
    time.sleep(1)
    vehicle.close()
    print("✓ 完成")

else:
    print("❌ 连接失败")