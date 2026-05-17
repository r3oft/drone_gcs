import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig
from dronekit import VehicleMode

logging.basicConfig(level=logging.DEBUG)

config = FlightConfig(connection_string="/dev/ttyUSB0", pixhawk_baud=57600)
fb = FlightBridge(config)

# 保存最新的 servo_raw 数据
latest_servo_raw = {'servo1': 0, 'servo2': 0, 'servo3': 0, 'servo4': 0}

# 保存最新的 LOCAL_POSITION_NED 数据
latest_local_pos = {'x': 0, 'y': 0, 'z': 0}

def servo_raw_callback(vehicle, name, msg):
    """捕获 SERVO_OUTPUT_RAW 消息"""
    global latest_servo_raw
    latest_servo_raw = {
        'servo1': msg.servo1_raw,
        'servo2': msg.servo2_raw,
        'servo3': msg.servo3_raw,
        'servo4': msg.servo4_raw,
    }

def local_position_ned_callback(vehicle, name, msg):
    """捕获 LOCAL_POSITION_NED 消息"""
    global latest_local_pos
    latest_local_pos = {
        'x': msg.x,
        'y': msg.y,
        'z': msg.z,
    }

print("测试：起飞流程")

if fb.connect():
    print("连接成功\n")
    vehicle = fb._vehicle
    
    # 注册 SERVO_OUTPUT_RAW 消息监听器
    vehicle.add_message_listener('SERVO_OUTPUT_RAW', servo_raw_callback)
    
    # 注册 LOCAL_POSITION_NED 消息监听器
    vehicle.add_message_listener('LOCAL_POSITION_NED', local_position_ned_callback)
    
    try:
        # 第一步：切换到 GUIDED 模式
        print("[1/3] 切换到 GUIDED 模式...")
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(2)
        print(f"✓ 模式: {vehicle.mode.name}\n")
        
        # 第二步：检查电机状态（遥控器已解锁）
        print("[2/3] 检查电机状态...")
        print(f"  电机: {'✓ 已解锁' if vehicle.armed else '❌ 未解锁'}")
        print(f"  需要遥控器手动解锁电机\n")
        
        if not vehicle.armed:
            print("⚠️  电机未解锁，请使用遥控器解锁后再继续")
            print("按 Enter 继续...")
            input()
        
        # 第三步：起飞
        print("[3/4] 起飞测试...")
        target_altitude = 0.3
        land_altitude = 0.3
        print(f"目标高度: {target_altitude}m\n")
        
        # 启动起飞
        print("发送起飞指令...")
        fb.arm_and_takeoff(target_altitude)

        print("悬停 10 秒...")
        time.sleep(10)

        # 第四步：测试 send_body_velocity
        print("[4/4] 测试 send_body_velocity...")
        start = time.time()
        while time.time() - start < 2.0:
            fb.send_body_velocity(0.2, 0.0, 0.0, 0.0)
            time.sleep(0.2)  # 5Hz

        fb.send_body_velocity(0.0, 0.0, 0.0, 0.0)
        print("\n停止速度指令")

        time.sleep(2)

        # print("发送降落指令")
        # if not fb.simple_goto(land_altitude):
        #     print("Altitude adjustment failed")
        print("发送降落指令...")
        fb.land()
        
        # print("监测高度和油门输出（按 Ctrl+C 停止）...\n")
        
        try:
            start_time = time.time()
            
            while True:
                try:
                    alt = latest_local_pos['z'] * -1  # NED 中 Z 向下为正，转换为正高度
                    elapsed = int(time.time() - start_time)
                    
                    # 打印到控制台
                    print(f"  [{elapsed}s] 高度: {alt:.2f}m | Servo: S1={latest_servo_raw['servo1']:4d} S2={latest_servo_raw['servo2']:4d} S3={latest_servo_raw['servo3']:4d} S4={latest_servo_raw['servo4']:4d}", end="\r")
                    
                except Exception as e:
                    print(f"\n❌ 读取高度失败: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n✓ 已停止监测")
        
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if vehicle.armed:
            print("\n上锁电机...")
            vehicle.armed = False
            time.sleep(1)
        
        vehicle.close()
        print("✓ 连接已关闭")
else:
    print("❌ 连接失败")
