import sys
import time
import logging
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig
from dronekit import VehicleMode
from pymavlink import mavutil

# 设置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EKF_DIAGNOSTICS")

# 创建诊断数据保存目录
diag_dir = Path(__file__).parent / "diagnostics"
diag_dir.mkdir(exist_ok=True)

# 诊断数据收集
ekf_data = {
    "timestamp": datetime.now().isoformat(),
    "samples": []
}

imu_data = {
    "timestamp": datetime.now().isoformat(),
    "samples": []
}

attitude_data = {
    "timestamp": datetime.now().isoformat(),
    "samples": []
}


def on_ekf_status(vehicle, name, msg):
    """处理 EKF_STATUS_REPORT 消息"""
    try:
        sample = {
            "time": time.time(),
            "flags": msg.flags,
            "vel_horiz": msg.vel_horiz,
            "vel_vert": msg.vel_vert,
            "pos_horiz_accuracy": msg.pos_horiz_accuracy,
            "pos_vert_accuracy": msg.pos_vert_accuracy,
            "terrain_alt_accuracy": msg.terrain_alt_accuracy,
            "compass_variation": msg.compass_variation,
        }
        ekf_data["samples"].append(sample)
        
        # 打印到控制台
        print(f"\n[EKF 状态]")
        print(f"  标志位: {msg.flags:#x}")
        print(f"  水平速度误差: {msg.vel_horiz:.3f} m/s")
        print(f"  竖直速度误差: {msg.vel_vert:.3f} m/s")
        print(f"  水平位置精度: {msg.pos_horiz_accuracy:.3f} m")
        print(f"  竖直位置精度: {msg.pos_vert_accuracy:.3f} m")
        
    except Exception as e:
        logger.error(f"处理 EKF_STATUS_REPORT 异常: {e}")


def on_raw_imu(vehicle, name, msg):
    """处理 RAW_IMU 消息 - 原始传感器数据"""
    try:
        sample = {
            "time": time.time(),
            "xacc": msg.xacc,  # X 加速度 (m/s^2 * 1000)
            "yacc": msg.yacc,
            "zacc": msg.zacc,
            "xgyro": msg.xgyro,  # X 陀螺速率 (rad/s * 1000)
            "ygyro": msg.ygyro,
            "zgyro": msg.zgyro,
            "xmag": msg.xmag,  # X 磁场 (gauss * 800)
            "ymag": msg.ymag,
            "zmag": msg.zmag,
            "temperature": msg.temperature if hasattr(msg, 'temperature') else None,
        }
        imu_data["samples"].append(sample)
        
        if len(imu_data["samples"]) % 10 == 0:  # 每10个样本打印一次
            print(f"\n[IMU 数据]")
            print(f"  加速度: X={msg.xacc/1000:.3f} Y={msg.yacc/1000:.3f} Z={msg.zacc/1000:.3f} m/s²")
            print(f"  陀螺仪: X={msg.xgyro/1000:.3f} Y={msg.ygyro/1000:.3f} Z={msg.zgyro/1000:.3f} rad/s")
            print(f"  磁力计: X={msg.xmag/800:.1f} Y={msg.ymag/800:.1f} Z={msg.zmag/800:.1f} gauss")
        
    except Exception as e:
        logger.error(f"处理 RAW_IMU 异常: {e}")


def on_attitude(vehicle, name, msg):
    """处理 ATTITUDE 消息"""
    try:
        sample = {
            "time": time.time(),
            "roll": msg.roll,      # 弧度
            "pitch": msg.pitch,    # 弧度
            "yaw": msg.yaw,        # 弧度
            "rollspeed": msg.rollspeed,
            "pitchspeed": msg.pitchspeed,
            "yawspeed": msg.yawspeed,
        }
        attitude_data["samples"].append(sample)
        
        # 转换为度
        roll_deg = msg.roll * 180 / 3.14159
        pitch_deg = msg.pitch * 180 / 3.14159
        yaw_deg = msg.yaw * 180 / 3.14159
        
        if len(attitude_data["samples"]) % 10 == 0:
            print(f"\n[姿态信息]")
            print(f"  滚转(Roll): {roll_deg:.2f}°")
            print(f"  俯仰(Pitch): {pitch_deg:.2f}°")
            print(f"  偏航(Yaw): {yaw_deg:.2f}°")
            print(f"  角速度: R={msg.rollspeed:.3f} P={msg.pitchspeed:.3f} Y={msg.yawspeed:.3f} rad/s")
        
    except Exception as e:
        logger.error(f"处理 ATTITUDE 异常: {e}")


def on_ahrs(vehicle, name, msg):
    """处理 AHRS 消息 - 姿态航向参考系统"""
    try:
        print(f"\n[AHRS 状态]")
        print(f"  滚转: {msg.roll * 180 / 3.14159:.2f}°")
        print(f"  俯仰: {msg.pitch * 180 / 3.14159:.2f}°")
        print(f"  偏航: {msg.yaw * 180 / 3.14159:.2f}°")
        print(f"  高度: {msg.altitude:.2f} m")
        print(f"  水平速度: {msg.v1:.2f}, {msg.v2:.2f} m/s")
        print(f"  竖直速度: {msg.v3:.2f} m/s")
    except Exception as e:
        logger.error(f"处理 AHRS 异常: {e}")


# 主程序
config = FlightConfig(connection_string="/dev/ttyUSB0", pixhawk_baud=57600)
fb = FlightBridge(config)

if fb.connect():
    print("✓ 连接成功\n")
    
    vehicle = fb._vehicle
    
    print("="*60)
    print("EKF3 诊断 - 开始收集数据")
    print("="*60)
    
    # 注册消息监听
    vehicle.add_message_listener('EKF_STATUS_REPORT', on_ekf_status)
    vehicle.add_message_listener('RAW_IMU', on_raw_imu)
    vehicle.add_message_listener('ATTITUDE', on_attitude)
    vehicle.add_message_listener('AHRS', on_ahrs)
    
    try:
        # 第一步：读取飞控参数
        print("\n" + "="*60)
        print("第一步：读取飞控参数")
        print("="*60)
        
        critical_params = [
            "AHRS_EKF_TYPE",
            "EK3_ENABLE",
            "EK3_SRC1_POSXY",
            "EK3_SRC1_VELXY",
            "EK3_SRC1_POSZ",
            "EK3_SRC1_VELZ",
            "INS_ACCEL_FILTER",
            "INS_GYRO_FILTER",
            "COMPASS_USE",
            "COMPASS_AUTODEC",
            "INS_ACC1CALX",
            "INS_ACC1CALY",
            "INS_ACC1CALZ",
            "INS_GYRO1CALX",
            "INS_GYRO1CALY",
            "INS_GYRO1CALZ",
        ]
        
        param_values = {}
        for param in critical_params:
            try:
                if param in vehicle.parameters:
                    value = vehicle.parameters[param]
                    param_values[param] = value
                    print(f"  {param}: {value}")
            except Exception as e:
                print(f"  ❌ {param}: 读取失败 - {e}")
        
        # 第二步：检查飞控就绪状态
        print("\n" + "="*60)
        print("第二步：检查飞控就绪状态")
        print("="*60)
        
        print(f"  is_armable: {vehicle.is_armable}")
        print(f"  system_status: {vehicle.system_status.state}")
        print(f"  armed: {vehicle.armed}")
        print(f"  mode: {vehicle.mode.name}")
        
        # 第三步：收集 30 秒的实时数据
        print("\n" + "="*60)
        print("第三步：收集实时数据（30秒）")
        print("="*60)
        print("\n监听中... 请注意控制台输出的 EKF、IMU、姿态数据\n")
        
        for i in range(30):
            print(f"  [{i+1}/30] 正在采集...", end="\r")
            time.sleep(1)
        
        print("\n✓ 数据采集完成\n")
        
        # 第四步：分析数据
        print("="*60)
        print("第四步：数据分析")
        print("="*60)
        
        if attitude_data["samples"]:
            rolls = [s["roll"] * 180 / 3.14159 for s in attitude_data["samples"]]
            pitches = [s["pitch"] * 180 / 3.14159 for s in attitude_data["samples"]]
            
            print(f"\n姿态统计：")
            print(f"  滚转: min={min(rolls):.2f}° max={max(rolls):.2f}° avg={sum(rolls)/len(rolls):.2f}°")
            print(f"  俯仰: min={min(pitches):.2f}° max={max(pitches):.2f}° avg={sum(pitches)/len(pitches):.2f}°")
            print(f"  ⚠️  如果 Roll/Pitch > 15°，说明飞机初始姿态有问题或IMU校准不准")
        
        if ekf_data["samples"]:
            print(f"\nEKF 统计：")
            print(f"  采集样本数: {len(ekf_data['samples'])}")
            vel_horiz = [s["vel_horiz"] for s in ekf_data["samples"]]
            print(f"  水平速度误差: min={min(vel_horiz):.3f} max={max(vel_horiz):.3f} avg={sum(vel_horiz)/len(vel_horiz):.3f} m/s")
        
        if imu_data["samples"]:
            print(f"\nIMU 统计：")
            print(f"  采集样本数: {len(imu_data['samples'])}")
            # 检查Z轴加速度是否接近 9.81
            z_accels = [s["zacc"]/1000 for s in imu_data["samples"]]
            print(f"  Z轴加速度: min={min(z_accels):.2f} max={max(z_accels):.2f} avg={sum(z_accels)/len(z_accels):.2f} m/s²")
            print(f"  ⚠️  静止时应约为 -9.81 m/s²（重力加速度）")
        
    finally:
        # 保存数据到文件
        print("\n" + "="*60)
        print("第五步：保存诊断数据")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ekf_file = diag_dir / f"ekf_data_{timestamp}.json"
        with open(ekf_file, 'w') as f:
            json.dump(ekf_data, f, indent=2)
        print(f"✓ EKF 数据已保存: {ekf_file}")
        
        imu_file = diag_dir / f"imu_data_{timestamp}.json"
        with open(imu_file, 'w') as f:
            json.dump(imu_data, f, indent=2)
        print(f"✓ IMU 数据已保存: {imu_file}")
        
        attitude_file = diag_dir / f"attitude_data_{timestamp}.json"
        with open(attitude_file, 'w') as f:
            json.dump(attitude_data, f, indent=2)
        print(f"✓ 姿态数据已保存: {attitude_file}")
        
        param_file = diag_dir / f"params_{timestamp}.json"
        with open(param_file, 'w') as f:
            json.dump(param_values, f, indent=2)
        print(f"✓ 参数已保存: {param_file}")
        
        # 生成诊断报告
        print("\n" + "="*60)
        print("诊断建议")
        print("="*60)
        
        if attitude_data["samples"]:
            rolls = [s["roll"] * 180 / 3.14159 for s in attitude_data["samples"]]
            pitches = [s["pitch"] * 180 / 3.14159 for s in attitude_data["samples"]]
            
            max_roll = max(abs(min(rolls)), abs(max(rolls)))
            max_pitch = max(abs(min(pitches)), abs(max(pitches)))
            
            if max_roll > 15 or max_pitch > 15:
                print("\n❌ 问题 1：飞机初始姿态异常（超过15°）")
                print("   原因可能:")
                print("   - IMU 加速度计未校准或校准不准")
                print("   - 飞机重心偏离")
                print("   - 飞机放置在不平的地面上")
                print("   解决方案:")
                print("   1. 在 QGC 中执行 IMU 校准（水平面上）")
                print("   2. 检查飞机重心位置")
                print("   3. 调试时将飞机放在平的地面上")
            else:
                print("\n✓ 飞机初始姿态正常（<15°）")
        
        print(f"\n更详细的诊断数据已保存到: {diag_dir}")
        print("可以用 Python 加载 JSON 文件进行离线分析")
        
        fb._vehicle.close()
        print("\n✓ 连接已关闭")

else:
    print("❌ 连接失败")