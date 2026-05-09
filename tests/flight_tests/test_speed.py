import sys
import time
import logging
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flight_bridge import FlightBridge, FlightConfig

# 设置更详细的日志
logging.basicConfig(level=logging.DEBUG)


config = FlightConfig(connection_string="/dev/ttyUSB0", pixhawk_baud=57600)
fb = FlightBridge(config)

if fb.connect():
    print("连接成功\n")
    
    # 启动后台监听线程（守护线程，脚本退出时自动关闭）
    monitor_thread = threading.Thread(target=monitor_rc_takeover, args=(fb,), daemon=True)
    monitor_thread.start()
    print("✓ 已启动遥控器接管监听\n")
    
    # 第一步：强制油门到最低，避免 failsafe
    print("=" * 60)
    print("第一步：准备环境")
    print("=" * 60)
    print("\n强制设置油门为最低...")
    fb._vehicle.channels.overrides['3'] = 1000
    time.sleep(1)
    
    # 第二步：切换到 GUIDED_NOGPS（支持速度指令）
    print("\n切换到 GUIDED 模式...")
    fb._vehicle.mode = "GUIDED"
    time.sleep(1)
    print(f"✓ 当前模式: {fb._vehicle.mode.name}\n")
    
    # 第三步：解锁
    print("解锁电机...")
    fb._vehicle.armed = True
    time.sleep(2)
    
    if fb._vehicle.armed and continue_test:
        print("✓ 电机已解锁\n")
        
        # 第四步：测试 send_body_velocity
        print("=" * 60)
        print("第二步：测试 send_body_velocity 函数")
        print("=" * 60)
        
        # 测试配置：[描述, (vx, vy, vz, yaw_rate)]
        test_cases = [
            ("零速度（悬停）", (0, 0, 0, 0)),
            ("向上 0.2 m/s", (0, 0, -0.2, 0)),
            ("向上 0.5 m/s（较快）", (0, 0, -0.5, 0)),
            ("向上 1.0 m/s（快速）", (0, 0, -1.0, 0)),
            ("前进 0.3 m/s", (0.3, 0, 0, 0)),
            ("向右 0.3 m/s", (0, 0.3, 0, 0)),
            ("顺时针转 0.5 rad/s", (0, 0, 0, 0.5)),
            ("复合：前进+向上", (0.3, 0, -0.3, 0)),
            ("回到悬停", (0, 0, 0, 0)),
        ]
        
        for desc, (vx, vy, vz, yaw_rate) in test_cases:
            # 检查是否需要停止测试
            if not continue_test:
                print(f"\n【中断】{takeover_reason}")
                break
            
            print(f"\n【{desc}】")
            print(f"  指令：vx={vx}, vy={vy}, vz={vz}, yaw_rate={yaw_rate}")
            
            # 连续发送 2 秒
            print("  发送中... 请观察电机转速变化")
            for i in range(10):
                # 再次检查是否需要立即停止
                if not continue_test:
                    break
                
                # 发送速度指令
                fb.send_body_velocity(vx, vy, vz, yaw_rate)
                time.sleep(0.2)
                
                # 读取并显示当前 PWM
                try:
                    ch = [
                        fb._vehicle.channels['1'],
                        fb._vehicle.channels['2'],
                        fb._vehicle.channels['3'],
                        fb._vehicle.channels['4']
                    ]
                    avg_ch = (ch[0] + ch[1] + ch[2] + ch[3]) / 4
                    print(f"    [{i+1}/10] PWM: {ch} (平均: {avg_ch:.0f})", end="\r")
                except Exception as e:
                    print(f"    ❌ 读取失败: {e}", end="\r")
            
            print()  # 换行
            time.sleep(1)
        
        # 第五步：恢复到零速度并上锁
        print("\n" + "=" * 60)
        print("第三步：清理")
        print("=" * 60)
        
        print("\n恢复到零速度...")
        for _ in range(5):
            fb.send_body_velocity(0, 0, 0, 0)
            time.sleep(0.2)
        
        time.sleep(2)
        
        # 如果是遥控器接管，返回手动模式；如果是正常完成，才上锁
        if continue_test:
            print("\n上锁电机...")
            fb._vehicle.armed = False
            time.sleep(1)
            
            if not fb._vehicle.armed:
                print("✓ 电机已上锁")
        else:
            print("\n⚠️  由于接管原因，保持电机状态让遥控器控制")
        
        # 诊断结果
        if continue_test:
            print("\n" + "=" * 60)
            print("测试完成 - 诊断结果")
            print("=" * 60)
            print("\n观察结果：")
            print("1. 向上速度：PWM 平均值应该逐步增加（从 ~1500 → ~1600+）")
            print("2. 前进速度：CH1/CH2 应该有变化")
            print("3. 转向速度：CH4 应该变化")
            print("4. 如果 PWM 有变化 → send_body_velocity 工作正常 ✓")
            print("5. 如果 PWM 没变化 → 可能需要调整参数或模式")
    
    # 停止监听线程（等待线程自然退出）
    continue_test = False
    time.sleep(0.5)
    
    fb._vehicle.close()