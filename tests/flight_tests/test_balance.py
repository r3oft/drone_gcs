import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig
from dronekit import VehicleMode, LocationGlobal

logging.basicConfig(level=logging.DEBUG)

config = FlightConfig(connection_string="/dev/ttyUSB0", pixhawk_baud=57600)
fb = FlightBridge(config)

# 保存最新的 servo_raw 数据
latest_servo_raw = {'servo1': 0, 'servo2': 0, 'servo3': 0, 'servo4': 0}

def servo_raw_callback(vehicle, name, msg):
    """捕获 SERVO_OUTPUT_RAW 消息"""
    global latest_servo_raw
    latest_servo_raw = {
        'servo1': msg.servo1_raw,
        'servo2': msg.servo2_raw,
        'servo3': msg.servo3_raw,
        'servo4': msg.servo4_raw,
    }

print("测试：起飞流程")

if fb.connect():
    print("连接成功\n")
    vehicle = fb._vehicle
    
    # 注册 SERVO_OUTPUT_RAW 消息监听器
    vehicle.add_message_listener('SERVO_OUTPUT_RAW', servo_raw_callback)
    
    try:
        
        # 打开文件保存数据
        log_file = Path(PROJECT_ROOT) / "servo_output.txt"
        print(f"监测油门输出（按 Ctrl+C 停止）...")
        print(f"数据保存到: {log_file}\n")
        
        try:
            start_time = time.time()
            with open(log_file, 'w') as f:
                # 写入文件头
                f.write("时间(s)\tRoll(°)\tPitch(°)\tYaw(°)\tServo1\tServo2\tServo3\tServo4\n")
                f.write("="*80 + "\n")
                
                while True:
                    try:
                        elapsed = int(time.time() - start_time)
                        
                        # 获取姿态数据（弧度转角度）
                        roll = vehicle.attitude.roll * 180 / 3.14159  # 弧度转度
                        pitch = vehicle.attitude.pitch * 180 / 3.14159
                        yaw = vehicle.attitude.yaw * 180 / 3.14159
                        
                        # 打印到控制台
                        print(f"  [{elapsed}s] R={roll:6.1f}° P={pitch:6.1f}° Y={yaw:6.1f}° | S1={latest_servo_raw['servo1']:4d} S2={latest_servo_raw['servo2']:4d} S3={latest_servo_raw['servo3']:4d} S4={latest_servo_raw['servo4']:4d}", end="\r")
                        
                        # 写入文件
                        f.write(f"{elapsed}\t{roll:.2f}\t{pitch:.2f}\t{yaw:.2f}\t{latest_servo_raw['servo1']}\t{latest_servo_raw['servo2']}\t{latest_servo_raw['servo3']}\t{latest_servo_raw['servo4']}\n")
                        f.flush()  # 实时保存
                    except:
                        pass
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n✓ 已停止监测")
            print(f"✓ 数据已保存到: {log_file}")
        
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()

else:
    print("❌ 连接失败")