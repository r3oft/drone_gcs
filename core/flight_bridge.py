from abc import ABC, abstractmethod
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass

# 导入 dronekit 核心模块
import dronekit
from dronekit import Vehicle, VehicleMode, APIException
from pymavlink import mavutil
import socket

from core.interfaces import IFlightBridge, IMCUBridge

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FlightBridge")

# 全局配置（可替换为配置文件读取）
@dataclass
class FlightConfig:
    """飞行控制配置项"""
    connection_string: str = "tcp:127.0.0.1:5760"  # 飞控连接串
    heartbeat_timeout: int = 15  # 连接心跳超时(s)
    takeoff_timeout_s: int = 30  # 起飞超时(s)
    land_timeout_s: int = 60  # 降落超时(s)
    land_detect_alt: float = 0.1  # 触地检测高度(m)
    pixhawk_baud: int = 115200  # Pixhawk 串口波特率
    mcu_serial_port: int = 4  # Pico 2 连接的 Pixhawk UART 端口
    mcu_baudrate: int = 115200  # Pico 2 波特率


class FlightBridge(IFlightBridge):
    """基于 DroneKit-Python 的飞控桥接实现"""

    def __init__(self, config: FlightConfig = None):
        self.config = config or FlightConfig()
        self._vehicle: Optional[Vehicle] = None
        self._last_heartbeat_time: float = 0.0

    def connect(self) -> bool:
        """
        建立与飞控器的 MAVLink 连接，阻塞等待首个心跳包
        """
        if self._vehicle is not None:
            logger.warning("已存在活跃连接，先断开旧连接")
            self._vehicle.close()

        try:
            # 建立连接（阻塞等待就绪）
            self._vehicle = dronekit.connect(
                self.config.connection_string,
                wait_ready=True,
                heartbeat_timeout=self.config.heartbeat_timeout,
                baud=self.config.pixhawk_baud   
            )
            # 注册心跳监听
            @self._vehicle.on_message('HEARTBEAT')
            def _on_heartbeat(self, name, msg):
                self._last_heartbeat_time = time.time()

            logger.info(f"飞控连接成功: {self.config.connection_string}")
            self._last_heartbeat_time = time.time()
            return True

        except socket.error:
            logger.error("飞控连接失败：无可用服务器/串口")
        except APIException:
            logger.error(f"飞控连接失败：心跳超时（{self.config.heartbeat_timeout}s）")
        except OSError as e:
            logger.error(f"飞控连接失败：串口错误 - {e}")
        except Exception as e:
            logger.error(f"飞控连接失败：未知错误 - {e}", exc_info=True)
        
        self._vehicle = None
        return False

    def arm_and_takeoff(self, target_alt: float) -> bool:
        """
        解锁电机 + 起飞到指定相对高度
        """
        if not self.is_connected():
            logger.error("无法执行起飞：飞控未连接")
            return False

        if target_alt <= 0:
            logger.error("目标高度必须大于0")
            return False

        start_time = time.time()
        vehicle = self._vehicle

        # 1. 等待飞控初始化完成
        logger.info("等待飞控初始化...")
        while not vehicle.is_armable:
            if time.time() - start_time > self.config.takeoff_timeout_s:
                logger.error("起飞超时：飞控未就绪")
                return False
            time.sleep(1)

        # 2. 设置 GUIDED 模式
        logger.info("切换到 GUIDED 模式...")
        vehicle.mode = VehicleMode("GUIDED")
        while vehicle.mode.name != "GUIDED":
            if time.time() - start_time > self.config.takeoff_timeout_s:
                logger.error("起飞超时：模式切换失败")
                return False
            time.sleep(0.5)

        # 3. 解锁电机
        logger.info("解锁电机...")
        vehicle.armed = True
        while not vehicle.armed:
            if time.time() - start_time > self.config.takeoff_timeout_s:
                logger.error("起飞超时：电机解锁失败")
                return False
            time.sleep(0.5)

        # 4. 执行起飞
        logger.info(f"起飞到目标高度：{target_alt}m")
        vehicle.simple_takeoff(target_alt)

        # 5. 等待到达目标高度
        while True:
            # 检查超时
            if time.time() - start_time > self.config.takeoff_timeout_s:
                logger.error(f"起飞超时：{self.config.takeoff_timeout_s}s 未到达目标高度")
                return False

            # 获取当前相对高度
            current_alt = vehicle.location.global_relative_frame.alt
            logger.debug(f"当前高度：{current_alt:.2f}m / 目标高度：{target_alt}m")

            # 高度达标（95%）则返回成功
            if current_alt >= target_alt * 0.95:
                logger.info(f"到达目标高度：{current_alt:.2f}m")
                return True

            time.sleep(0.5)

    def send_body_velocity(
        self, vx: float, vy: float, vz: float, yaw_rate: float
    ) -> None:
        """
        发送机体坐标系速度指令（MAV_FRAME_BODY_NED）
        type_mask=0x07C7 → 仅控制速度和偏航角速度
        """
        if not self.is_connected():
            logger.error("无法发送速度指令：飞控未连接")
            return

        vehicle = self._vehicle
        # 构造 SET_POSITION_TARGET_LOCAL_NED 消息
        msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0,  # 时间戳（未使用）
            0, 0,  # 目标系统/组件
            mavutil.mavlink.MAV_FRAME_BODY_NED,  # 机体坐标系（NED）
            0x07C7,  # type_mask：忽略位置/加速度/合力，仅保留速度+偏航角速度
            0, 0, 0,  # x/y/z 位置（忽略）
            vx, vy, vz,  # x/y/z 速度（m/s）
            0, 0, 0,  # x/y/z 加速度（忽略）
            0, yaw_rate  # 偏航角、偏航角速度（rad/s）
        )
        # 发送消息（非阻塞）
        vehicle.send_mavlink(msg)
        vehicle.flush()
        logger.debug(f"发送速度指令：vx={vx}, vy={vy}, vz={vz}, yaw_rate={yaw_rate}")

    def land(self) -> bool:
        """
        切换到 LAND 模式并等待触地
        """
        if not self.is_connected():
            logger.error("无法执行降落：飞控未连接")
            return False

        start_time = time.time()
        vehicle = self._vehicle

        # 1. 切换到 LAND 模式
        logger.info("切换到 LAND 模式...")
        vehicle.mode = VehicleMode("LAND")
        while vehicle.mode.name != "LAND":
            if time.time() - start_time > self.config.land_timeout_s:
                logger.error("降落超时：模式切换失败")
                return False
            time.sleep(0.5)

        # 2. 等待触地
        logger.info("等待降落触地...")
        while True:
            # 检查超时
            if time.time() - start_time > self.config.land_timeout_s:
                logger.error(f"降落超时：{self.config.land_timeout_s}s 未触地")
                return False

            # 条件1：飞控解除解锁状态
            if not vehicle.armed:
                logger.info("飞控已解锁，确认触地")
                return True

            # 条件2：相对高度低于触地检测阈值
            current_alt = vehicle.location.global_relative_frame.alt
            if current_alt < self.config.land_detect_alt:
                logger.info(f"高度低于触地阈值（{current_alt:.2f}m），确认触地")
                return True

            time.sleep(0.5)

    def set_mode(self, mode: str) -> bool:
        """
        切换飞行模式
        """
        valid_modes = ["GUIDED", "LOITER", "RTL"]
        if mode not in valid_modes:
            logger.error(f"无效模式：{mode}，仅支持 {valid_modes}")
            return False

        if not self.is_connected():
            logger.error("无法切换模式：飞控未连接")
            return False

        try:
            # 设置模式
            self._vehicle.mode = VehicleMode(mode)
            # 等待模式切换完成
            timeout = 5  # 模式切换超时5s
            start_time = time.time()
            while self._vehicle.mode.name != mode:
                if time.time() - start_time > timeout:
                    logger.error(f"模式切换超时：{mode}")
                    return False
                time.sleep(0.2)

            logger.info(f"模式切换成功：{mode}")
            return True
        except APIException as e:
            logger.error(f"模式切换失败：{e}")
            return False

    def get_telemetry(self) -> dict:
        """
        获取当前飞行遥测数据
        """
        if not self.is_connected():
            return {
                "armed": False,
                "mode": "",
                "alt": 0.0,
                "heading": 0.0,
                "battery_pct": 0.0,
                "heartbeat_ok": False
            }

        vehicle = self._vehicle
        # 构造遥测数据
        telemetry = {
            "armed": vehicle.armed,
            "mode": vehicle.mode.name,
            "alt": vehicle.location.global_relative_frame.alt,
            "heading": vehicle.heading,  # 航向角（度，0-360）
            "battery_pct": vehicle.battery.level / 100.0 if vehicle.battery.level else 0.0,
            "heartbeat_ok": self.is_connected()
        }
        return telemetry

    def is_connected(self) -> bool:
        """
        检查飞控连接是否存活（心跳未超时）
        """
        if self._vehicle is None:
            return False

        # 检查心跳超时（允许2倍心跳超时时间）
        heartbeat_expired = (time.time() - self._last_heartbeat_time) > (self.config.heartbeat_timeout * 2)
        if heartbeat_expired:
            logger.warning("飞控心跳超时，连接已断开")
            self._vehicle = None
            return False

        return True


class MCUBridge(IMCUBridge):
    """
    末端执行器（Pico 2）通信桥接实现
    基于 MAVLink SERIAL_CONTROL 指令与 Pixhawk 串口通信
    """

    def __init__(self, flight_bridge: FlightBridge, config: FlightConfig = None):
        self.config = config or FlightConfig()
        self._flight_bridge = flight_bridge
        self._vehicle: Optional[Vehicle] = None
        self._listener_vehicle: Optional[Vehicle] = None
        self._response_buffer: Optional[str] = None
        self._response_map = {
            b"GRAB_DONE": "GRAB_DONE",
            b"GRAB_FAIL": "GRAB_FAIL",
            b"RELEASE_DONE": "RELEASE_DONE",
            b"RELEASE_FAIL": "RELEASE_FAIL",
            b"RESET_DONE": "RESET_DONE"
        }

        # 监听回调函数需要保持同一引用，便于重连后重新绑定。
        self._serial_listener = self._build_serial_listener()

        # 注册 SERIAL_CONTROL 消息监听
        self._register_serial_listener()

    def _build_serial_listener(self):
        """构建串口数据监听回调。"""
        def _on_serial_control(vehicle, name, msg):
            """处理从 Pixhawk 串口收到的 Pico 2 响应"""
            if msg.port != self.config.mcu_serial_port:
                return

            # 解析串口数据
            data = bytes(msg.data[:msg.count])
            for resp_bytes, resp_str in self._response_map.items():
                if resp_bytes in data:
                    self._response_buffer = resp_str
                    logger.debug(f"收到 Pico 2 响应：{resp_str}")
                    break

        return _on_serial_control

    def _register_serial_listener(self):
        """注册串口数据监听回调，必要时重新绑定到最新飞控连接。"""
        current_vehicle = self._flight_bridge._vehicle
        if current_vehicle is None:
            self._vehicle = None
            self._listener_vehicle = None
            return False

        # 同一连接对象已绑定过监听，无需重复注册。
        if self._listener_vehicle is current_vehicle:
            self._vehicle = current_vehicle
            return True

        self._vehicle = current_vehicle
        self._vehicle.add_message_listener('SERIAL_CONTROL', self._serial_listener)
        self._listener_vehicle = current_vehicle
        return True

    def send_command(self, command: str) -> bool:
        """
        向 Pico 2 发送控制指令
        通过 MAVLink SERIAL_CONTROL 指令转发到 Pixhawk UART4
        """
        valid_commands = ["START_GRAB", "START_RELEASE", "RESET"]
        if command not in valid_commands:
            logger.error(f"无效MCU指令：{command}，仅支持 {valid_commands}")
            return False

        if not self._flight_bridge.is_connected():
            logger.error("无法发送MCU指令：飞控未连接")
            return False

        # 每次发送前对齐最新飞控连接，避免重连后引用旧 vehicle。
        if not self._register_serial_listener():
            logger.error("无法发送MCU指令：飞控连接对象不可用")
            return False

        try:
            # 构造 SERIAL_CONTROL 消息
            cmd_bytes = command.encode('ascii')
            msg = self._vehicle.message_factory.serial_control_encode(
                self.config.mcu_serial_port,  # 目标串口
                0,  # 预留
                0,  # 操作：写入
                self.config.mcu_baudrate,  # 波特率
                len(cmd_bytes),  # 数据长度
                cmd_bytes  # 指令数据
            )
            # 发送消息
            self._vehicle.send_mavlink(msg)
            self._vehicle.flush()
            logger.info(f"发送MCU指令：{command}")
            return True
        except Exception as e:
            logger.error(f"发送MCU指令失败：{e}", exc_info=True)
            return False

    def get_latest_response(self) -> str | None:
        """非阻塞读取 Pico 2 的最新响应"""
        if self._response_buffer is None:
            return None

        # 取出并清空缓冲区（确保每次只返回一次）
        resp = self._response_buffer
        self._response_buffer = None
        return resp

    def is_connected(self) -> bool:
        """检查 MCU 通信链路是否存活"""
        # MCU 链路依赖飞控连接，且必须与当前飞控连接对象一致。
        current_vehicle = self._flight_bridge._vehicle
        return (
            self._flight_bridge.is_connected()
            and self._vehicle is not None
            and self._vehicle is current_vehicle
        )

