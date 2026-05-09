import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig
from dronekit import VehicleMode

config = FlightConfig(connection_string="/dev/ttyUSB0", pixhawk_baud=57600)
fb = FlightBridge(config)

if fb.connect():
    vehicle = fb._vehicle
    
    print("="*60)
    print("EKF 状态检查")
    print("="*60)
    
    # 1. 检查 IMU 数据
    print("\n1️⃣  IMU 加速度计校准检查")
    print("-" * 40)
    
    z_accels = []
    
    def on_raw_imu(v, name, msg):
        z_accels.append(msg.zacc / 1000.0)
    
    vehicle.add_message_listener('RAW_IMU', on_raw_imu)
    
    print("采集 5 秒 IMU 数据...")
    for i in range(5):
        print(f"  [{i+1}/5]", end="\r")
        time.sleep(1)
    
    if z_accels:
        avg_z = sum(z_accels) / len(z_accels)
        print(f"\nZ 轴加速度平均值: {avg_z:.2f} m/s²")
        
        if abs(avg_z + 9.81) < 0.5:
            print("✅ IMU 校准正常（-9.81 ± 0.5 m/s²）")
        else:
            print(f"❌ IMU 校准异常（应该是 -9.81，现在是 {avg_z:.2f}）")
            print("   需要在 QGC 中重新执行 IMU 加速度计校准")
    
    # 2. 检查 EKF 参数配置
    print("\n2️⃣  EKF 参数检查")
    print("-" * 40)
    
    ekf_params = {
        "AHRS_EKF_TYPE": 3,
        "EK3_ENABLE": 1,
        "EK3_SRC1_POSXY": 4,
        "EK3_SRC1_VELXY": 4,
        "EK3_SRC1_POSZ": 1,
        "EK3_SRC1_VELZ": 1,
    }
    
    all_correct = True
    for param_name, expected in ekf_params.items():
        if param_name in vehicle.parameters:
            actual = vehicle.parameters[param_name]
            status = "✅" if actual == expected else "❌"
            print(f"{status} {param_name}: {actual} (expected: {expected})")
            if actual != expected:
                all_correct = False
    
    # 3. 尝试切换到 GUIDED 看是否能通过 PreArm
    print("\n3️⃣  PreArm 检查（尝试进入 GUIDED 模式）")
    print("-" * 40)
    
    print(f"当前模式: {vehicle.mode.name}")
    print(f"当前 is_armable: {vehicle.is_armable}")
    
    if not vehicle.armed and vehicle.mode.name != "GUIDED":
        print("\n切换到 GUIDED 模式...")
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(2)
        
        print(f"模式已切换: {vehicle.mode.name}")
        print(f"is_armable: {vehicle.is_armable}")
        
        if vehicle.is_armable:
            print("✅ 可以进入 GUIDED 模式 - EKF 检查通过")
        else:
            print("❌ 无法进入 GUIDED 模式 - 检查飞控日志中的 PreArm 错误")
    
    # 总结
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    
    if all_correct and vehicle.is_armable:
        print("\n✅ EKF 配置正确，IMU 校准正常")
        print("   可以进行起飞测试了")
    else:
        print("\n❌ 仍有问题需要解决")
        print("   请检查上面的报告")
    
    vehicle.close()
else:
    print("连接失败")