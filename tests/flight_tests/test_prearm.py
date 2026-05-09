import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig
from dronekit import VehicleMode

logging.basicConfig(level=logging.DEBUG)

config = FlightConfig(connection_string="/dev/ttyUSB0", pixhawk_baud=57600)
fb = FlightBridge(config)

if fb.connect():
    print("连接成功\n")
    vehicle = fb._vehicle
    
    # 收集飞控日志
    flight_logs = []
    
    def on_statustext(vehicle, name, msg):
        """捕获飞控所有日志消息"""
        try:
            # msg.text 是字符串，msg.severity 是日志级别
            severity = msg.severity
            text = msg.text
            
            flight_logs.append({
                'severity': severity,
                'text': text,
                'time': time.time()
            })
            
            # 实时打印 CRITICAL 和 WARNING
            if severity <= 2:  # CRITICAL(0) 或 ERROR(1)
                print(f"[CRITICAL] {text}")
            elif severity == 3:  # WARNING
                print(f"[WARNING] {text}")
            elif "PreArm" in text:
                print(f"[PreArm] {text}")
                
        except Exception as e:
            print(f"日志处理异常: {e}")
    
    print("="*60)
    print("监听飞控状态 - 捕获 PreArm 错误")
    print("="*60 + "\n")
    
    # 注册日志监听
    vehicle.add_message_listener('STATUSTEXT', on_statustext)
    
    try:
        # 第一步：准备
        print("[1/4] 设置油门...")
        vehicle.channels.overrides['3'] = 1100
        time.sleep(1)
        print("✓ 油门已设置\n")
        
        # 第二步：切换模式
        print("[2/4] 切换到 GUIDED 模式...")
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(2)
        print(f"✓ 模式: {vehicle.mode.name}\n")
        
        # 第三步：监听 30 秒，等待飞控状态变化
        print("[3/4] 监听飞控日志 (30秒)...")
        print("-" * 40)
        
        start_time = time.time()
        while time.time() - start_time < 120:
            try:
                is_armable = vehicle.is_armable
                status = vehicle.system_status.state
                mode = vehicle.mode.name
                
                print(f"  is_armable={is_armable} | status={status} | mode={mode}", end="\r")
                
            except Exception as e:
                print(f"  读取状态异常: {e}", end="\r")
            
            time.sleep(0.5)
        
        print("\n\n[4/4] 分析结果")
        print("-" * 40)
        
        # 分析收集到的日志
        print(f"\n捕获到 {len(flight_logs)} 条飞控日志:\n")
        
        prearmed_errors = []
        for log in flight_logs:
            if "PreArm" in log['text']:
                prearmed_errors.append(log['text'])
                print(f"❌ {log['text']}")
        
        if not prearmed_errors:
            print("✓ 没有 PreArm 错误")
        
        # 最终状态
        print("\n" + "="*60)
        print("诊断总结")
        print("="*60)
        
        print(f"\n飞控状态:")
        print(f"  is_armable: {vehicle.is_armable}")
        print(f"  system_status: {vehicle.system_status.state}")
        print(f"  mode: {vehicle.mode.name}")
        print(f"  armed: {vehicle.armed}")
        
        if vehicle.is_armable:
            print("\n✅ 飞控已就绪，可以起飞")
        else:
            print("\n❌ 飞控未就绪，原因：")
            if prearmed_errors:
                for err in prearmed_errors:
                    print(f"   - {err}")
            else:
                print("   （未捕获到具体错误，可能是其他原因）")
                print("   请检查：")
                print("     1. IMU 校准是否完成")
                print("     2. EKF 参数是否正确")
                print("     3. 磁罗盘是否校准")
                print("     4. GPS/光流数据是否有效")
        
    except Exception as e:
        print(f"\n❌ 异常: {e}", exc_info=True)
    
    finally:
        vehicle.close()
        print("\n✓ 连接已关闭")
else:
    print("❌ 连接失败")