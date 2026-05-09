#!/usr/bin/env python3
"""
速度控制修复测试脚本

测试多种配置：
1. 不同的坐标系（BODY_NED, BODY_OFFSET_NED, LOCAL_NED）
2. 不同的 type_mask 值
3. 不同的飞行模式（GUIDED vs GUIDED_NOGPS）
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig
from pymavlink import mavutil

print("=" * 70)
print("速度控制修复测试脚本")
print("=" * 70)

config = FlightConfig(connection_string="/dev/ttyUSB0", pixhawk_baud=57600)
fb = FlightBridge(config)

if not fb.connect():
    print("❌ 连接失败")
    sys.exit(1)

vehicle = fb._vehicle

print("\n【准备环境】")
print("-" * 70)

# 强制油门最低
print("  设置油门最低...")
vehicle.channels.overrides['3'] = 1000
time.sleep(1)

# 切换到 GUIDED_NOGPS
print("  切换到 GUIDED_NOGPS...")
vehicle.mode = "GUIDED_NOGPS"
time.sleep(1)

# 解锁
print("  解锁电机...")
vehicle.armed = True
time.sleep(2)

if not vehicle.armed:
    print("  ❌ 解锁失败")
    vehicle.close()
    sys.exit(1)

print("  ✓ 准备完成")

# 测试配置列表
test_configs = [
    {
        "name": "配置 1: BODY_OFFSET_NED + 0x0FC7",
        "frame": mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        "type_mask": 0x0FC7,
        "vz": -0.5,
    },
    {
        "name": "配置 2: BODY_NED + 0x0FC7",
        "frame": mavutil.mavlink.MAV_FRAME_BODY_NED,
        "type_mask": 0x0FC7,
        "vz": -0.5,
    },
    {
        "name": "配置 3: BODY_OFFSET_NED + 0x07C7",
        "frame": mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        "type_mask": 0x07C7,
        "vz": -0.5,
    },
    {
        "name": "配置 4: LOCAL_NED + 0x0FC7",
        "frame": mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        "type_mask": 0x0FC7,
        "vz": -0.5,
    },
]

print("\n【测试速度控制】")
print("-" * 70)

for i, cfg in enumerate(test_configs, 1):
    print(f"\n  [{i}/{len(test_configs)}] {cfg['name']}")

    # 记录初始 PWM
    ch_before = vehicle.channels['3']
    print(f"      初始油门 PWM: {ch_before}")

    # 发送速度指令（连续 2 秒）
    print(f"      发送速度指令: vz={cfg['vz']} m/s")
    for j in range(10):
        msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            cfg['frame'],
            cfg['type_mask'],
            0, 0, 0,
            0, 0, cfg['vz'],
            0, 0, 0,
            0, 0
        )
        vehicle.send_mavlink(msg)
        vehicle.flush()
        time.sleep(0.2)

    # 记录最终 PWM
    ch_after = vehicle.channels['3']
    print(f"      最终油门 PWM: {ch_after}")

    # 判断效果
    delta = ch_after - ch_before
    if abs(delta) > 10:
        print(f"      ✅ 有效！PWM 变化: {delta:+d}")
        print(f"      >>> 找到有效配置！<<<")
        break
    else:
        print(f"      ❌ 无效，PWM 变化: {delta:+d}")

    # 恢复油门
    vehicle.channels.overrides['3'] = 1000
    time.sleep(0.5)

print("\n【测试不同飞行模式】")
print("-" * 70)

# 尝试切换到 GUIDED 模式
print("\n  切换到 GUIDED 模式...")
try:
    vehicle.mode = "GUIDED"
    time.sleep(1)

    if vehicle.mode.name == "GUIDED":
        print("  ✓ 切换成功")

        # 使用最佳配置重新测试
        best_cfg = test_configs[0]  # BODY_OFFSET_NED + 0x0FC7

        print(f"\n  使用 {best_cfg['name']} 重新测试")
        ch_before = vehicle.channels['3']
        print(f"      初始油门 PWM: {ch_before}")

        for j in range(10):
            msg = vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0,
                best_cfg['frame'],
                best_cfg['type_mask'],
                0, 0, 0,
                0, 0, best_cfg['vz'],
                0, 0, 0,
                0, 0
            )
            vehicle.send_mavlink(msg)
            vehicle.flush()
            time.sleep(0.2)

        ch_after = vehicle.channels['3']
        print(f"      最终油门 PWM: {ch_after}")
        delta = ch_after - ch_before

        if abs(delta) > 10:
            print(f"      ✅ GUIDED 模式下有效！PWM 变化: {delta:+d}")
        else:
            print(f"      ❌ GUIDED 模式下仍无效，PWM 变化: {delta:+d}")
    else:
        print(f"  ❌ 切换失败，当前模式: {vehicle.mode.name}")

except Exception as e:
    print(f"  ❌ 切换 GUIDED 模式失败: {e}")

print("\n【清理】")
print("-" * 70)
vehicle.channels.overrides['3'] = 1000
time.sleep(1)
vehicle.armed = False
time.sleep(1)
vehicle.close()
print("✓ 完成")

print("\n【诊断结论】")
print("-" * 70)
print("""
如果所有配置都无效，可能的原因：

1. 固件版本过旧（ArduCopter 3.6.7）
   → 解决方案：升级到 ArduCopter 4.x

2. 飞控参数未配置
   → 检查 Mission Planner 中的参数：
     - GUIDED_OPTIONS
     - WPNAV_SPEED
     - WPNAV_SPEED_UP
     - WPNAV_SPEED_DN

3. 飞控不支持速度控制
   → 解决方案：使用 channels.overrides 替代

4. 需要 GPS 定位
   → 解决方案：在室外测试或使用光流传感器

建议：
✅ 如果速度控制不工作，使用 channels.overrides 是可靠的替代方案
✅ 升级飞控固件到 ArduCopter 4.x 以获得更好的支持
""")
