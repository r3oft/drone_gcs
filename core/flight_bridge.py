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
    heartbeat_timeout: int = 60  # 连接心跳超时(s)
    takeoff_timeout_s: int = 30  # 起飞超时(s)
    land_timeout_s: int = 60  # 降落超时(s)
    land_detect_alt: float = 0.1  # 触地检测高度(m)
    pixhawk_baud: int = 57600  # Pixhawk 串口波特率
    mcu_serial_port: int = 4  # Pico 2 连接的 Pixhawk UART 端口
    mcu_baudrate: int = 115200  # Pico 2 波特率
    goto_vertical_speed: float = 0.15
    goto_alt_tolerance: float = 0.05
    goto_command_hz: float = 5.0
    local_position_timeout_s: float = 2.0
    reconnect_enabled: bool = False
    reconnect_max_attempts: int = 3
    reconnect_backoff_s: float = 1.0


class FlightBridge(IFlightBridge):
    """基于 DroneKit-Python 的飞控桥接实现"""

    def __init__(self, config: FlightConfig = None):
        self.config = config or FlightConfig()
        self._vehicle: Optional[Vehicle] = None
        self._last_heartbeat_time: float = 0.0
        self._latest_local_position_ned: Optional[Dict[str, float]] = None
        self._latest_local_position_time: float = 0.0

    def connect(self) -> bool:
        """
        建立与飞控器的 MAVLink 连接，阻塞等待首个心跳包
        """
        if self._vehicle is not None:
            logger.warning("已存在活跃连接，先断开旧连接")
            self._vehicle.close()

        try:
            # 连接，拉取参数
            logger.info(f"正在连接: {self.config.connection_string}")
            self._vehicle = dronekit.connect(
                self.config.connection_string,
                wait_ready=False,  # 改成 False
                heartbeat_timeout=self.config.heartbeat_timeout,
                baud=self.config.pixhawk_baud
            )

            logger.info("心跳已就绪，开始拉取参数...")

            try:
                logger.info("尝试拉取参数...")
                self._vehicle.wait_ready(timeout=120)
                logger.info("参数拉取成功")
            except Exception as param_err:
                logger.warning(f"参数拉取失败，但继续使用：{param_err}")

            # 注册心跳监听
            @self._vehicle.on_message('HEARTBEAT')
            def _on_heartbeat(vehicle, name, msg):
                self._last_heartbeat_time = time.time()

            @self._vehicle.on_message('LOCAL_POSITION_NED')
            def _on_local_position_ned(vehicle, name, msg):
                self._record_local_position_ned(msg)

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

    def _record_local_position_ned(self, msg) -> None:
        """Cache LOCAL_POSITION_NED so altitude control can avoid global coordinates."""
        try:
            self._latest_local_position_ned = {
                "x": float(msg.x),
                "y": float(msg.y),
                "z": float(msg.z),
                "vx": float(getattr(msg, "vx", 0.0)),
                "vy": float(getattr(msg, "vy", 0.0)),
                "vz": float(getattr(msg, "vz", 0.0)),
                "time_boot_ms": float(getattr(msg, "time_boot_ms", 0.0)),
            }
            self._latest_local_position_time = time.time()
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug(f"Ignoring invalid LOCAL_POSITION_NED message: {exc}")

    def _get_local_altitude(self) -> Optional[float]:
        """
        Return altitude above the EKF origin using LOCAL_POSITION_NED.

        MAVLink LOCAL_POSITION_NED uses NED coordinates, so z/down is positive
        downward. Height above the origin is therefore -z.
        """
        if self._latest_local_position_ned is not None:
            age = time.time() - self._latest_local_position_time
            if age <= self.config.local_position_timeout_s:
                return -float(self._latest_local_position_ned["z"])
            logger.debug(f"LOCAL_POSITION_NED data is stale: {age:.2f}s old")
            return None

        if self._vehicle is None:
            return None

        location = getattr(self._vehicle, "location", None)
        local_frame = getattr(location, "local_frame", None)
        down = getattr(local_frame, "down", None)
        if down is None:
            return None
        return -float(down)

    def _get_altitude(self) -> Optional[float]:
        """Read altitude with LOCAL_POSITION_NED first, global-relative only as fallback."""
        local_alt = self._get_local_altitude()
        if local_alt is not None:
            return local_alt

        if self._vehicle is None:
            return None

        location = getattr(self._vehicle, "location", None)
        global_relative_frame = getattr(location, "global_relative_frame", None)
        global_alt = getattr(global_relative_frame, "alt", None)
        if global_alt is None:
            return None
        return float(global_alt)

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
        # while not vehicle.is_armable:
        #     if time.time() - start_time > self.config.takeoff_timeout_s:
        #         logger.error("起飞超时：飞控未就绪")
        #         return False
        #     time.sleep(1)

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

        # while True:
        #     if time.time() - start_time > self.config.takeoff_timeout_s:
        #         logger.error(f"takeoff timed out after {self.config.takeoff_timeout_s}s")
        #         return False

        # #     current_alt = float(vehicle.location.global_relative_frame.alt or 0.0)
        # #     logger.debug(f"当前高度：{current_alt:.2f}m / 目标高度：{target_alt}m")

        #     logger.debug(f"Current altitude: {current_alt:.2f}m / target: {target_alt:.2f}m")
        #     if current_alt >= target_alt * 0.95:
        #         logger.info(f"Reached takeoff altitude: {current_alt:.2f}m")
        #         return True

        #     time.sleep(0.5)

    def send_body_velocity(
        self, vx: float, vy: float, vz: float, yaw_rate: float
    ) -> None:
        """
        发送机体坐标系速度指令（MAV_FRAME_BODY_NED）

        type_mask 位定义（bit=1 表示忽略该字段）：
        bit 0-2: 位置 x, y, z
        bit 3-5: 速度 vx, vy, vz
        bit 6-8: 加速度 ax, ay, az
        bit 9:   force
        bit 10:  偏航角 yaw
        bit 11:  偏航角速度 yaw_rate

        0x07C7 = 0000 0111 1100 0111
        = 忽略位置(0-2)、加速度(6-8)、force(9)、偏航角(10)
        = 仅使用速度(3-5) + 偏航角速度(11)

        注意：ArduCopter 3.6.x 对 MAV_FRAME_BODY_NED 支持有限，
        如果不工作，建议升级到 4.x 或使用 MAV_FRAME_BODY_OFFSET_NED
        """
        if not self.is_connected():
            logger.error("无法发送速度指令：飞控未连接")
            return

        vehicle = self._vehicle

        # 检查飞行模式（速度控制需要 GUIDED 模式）
        if vehicle.mode.name not in ["GUIDED", "GUIDED_NOGPS"]:
            logger.warning(f"当前模式 {vehicle.mode.name} 可能不支持速度控制，建议切换到 GUIDED")

        # 构造 SET_POSITION_TARGET_LOCAL_NED 消息
        # 参考：https://mavlink.io/en/messages/common.html#SET_POSITION_TARGET_LOCAL_NED
        msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0,  # time_boot_ms (not used)
            0, 0,  # target system, target component
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # 使用 BODY_OFFSET_NED（兼容性更好）
            0x07C7,  # type_mask: 仅使用速度 + 偏航角速度
            0, 0, 0,  # x, y, z positions (ignored)
            vx, vy, vz,  # x, y, z velocity in m/s
            0, 0, 0,  # x, y, z acceleration (ignored)
            0,       # yaw (ignored)
            yaw_rate # yaw_rate in rad/s
        )
        # 发送消息（非阻塞）
        vehicle.send_mavlink(msg)
        vehicle.flush()
        logger.debug(f"发送速度指令：vx={vx}, vy={vy}, vz={vz}, yaw_rate={yaw_rate}")

    def send_attitude_target(
        self,
        roll_rate: float,
        pitch_rate: float,
        yaw_rate: float,
        thrust: float
    ) -> None:
        """
        发送姿态控制（推荐：角速度 + thrust）

        参数：
        - roll_rate, pitch_rate, yaw_rate: rad/s
        - thrust: 0~1（0.5≈悬停）
        """

        if not self.is_connected():
            logger.error("飞控未连接")
            return

        vehicle = self._vehicle

        if vehicle.mode.name not in ["GUIDED", "GUIDED_NOGPS"]:
            logger.warning(f"当前模式 {vehicle.mode.name} 不适合姿态控制")

        try:
            # 忽略姿态，仅使用角速度
            type_mask = 0b00000111  # = 7

            msg = vehicle.message_factory.set_attitude_target_encode(
                0,      # time_boot_ms
                0, 0,   # target system/component
                type_mask,
                [1, 0, 0, 0],  # 必填，但会被忽略
                roll_rate,
                pitch_rate,
                yaw_rate,
                thrust
            )

            vehicle.send_mavlink(msg)
            vehicle.flush()

        except Exception as e:
            logger.error(f"发送姿态指令失败: {e}", exc_info=True)

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

        logger.info("Waiting for landing touchdown...")
        while True:
            if time.time() - start_time > self.config.land_timeout_s:
                logger.error(f"Landing timed out after {self.config.land_timeout_s}s")
                return False

            if not vehicle.armed:
                logger.info("Vehicle disarmed; landing confirmed")
                return True

            current_alt = self._get_local_altitude()
            if current_alt is None:
                logger.debug("LOCAL_POSITION_NED unavailable; falling back to global-relative altitude")
                current_alt = self._get_altitude()
            if current_alt is not None and current_alt < self.config.land_detect_alt:
                logger.info(f"Altitude below land threshold: {current_alt:.2f}m")
                return True

            time.sleep(0.5)

    def simple_goto(self, target_alt: float) -> bool:
        """
        维持当前位置，仅改变相对高度（基于 simple_goto）
        """
        if not self.is_connected():
            logger.error("无法执行 simple_goto：飞控未连接")
            return False

        if target_alt < 0:
            logger.error("目标高度必须大于等于0")
            return False

        return self._simple_goto_velocity(target_alt)

    def _simple_goto_velocity(self, target_alt: float) -> bool:
        """Adjust relative altitude using streamed vertical velocity commands."""
        vehicle = self._vehicle
        if vehicle.mode.name not in ["GUIDED", "GUIDED_NOGPS"]:
            logger.info("Switching to GUIDED before altitude adjustment")
            if not self.set_mode("GUIDED"):
                return False

        speed = max(0.01, abs(self.config.goto_vertical_speed))
        tolerance = max(0.01, abs(self.config.goto_alt_tolerance))
        command_hz = max(1.0, self.config.goto_command_hz)
        command_interval = 1.0 / command_hz

        logger.info(f"Adjusting altitude to {target_alt:.2f}m with vertical velocity")
        start_time = time.time()
        while True:
            if time.time() - start_time > self.config.takeoff_timeout_s:
                logger.error("simple_goto timed out before reaching target altitude")
                self.send_body_velocity(0, 0, 0, 0)
                return False

            current_alt = self._get_altitude()
            if current_alt is None:
                logger.error("simple_goto cannot read LOCAL_POSITION_NED altitude")
                self.send_body_velocity(0, 0, 0, 0)
                return False

            error = target_alt - float(current_alt)
            if abs(error) <= tolerance:
                self.send_body_velocity(0, 0, 0, 0)
                logger.info(f"simple_goto reached altitude: {current_alt:.2f}m")
                return True

            vz = -speed if error > 0 else speed
            self.send_body_velocity(0, 0, vz, 0)
            time.sleep(command_interval)

    def set_mode(self, mode: str) -> bool:
        """
        切换飞行模式
        """
        valid_modes = ["GUIDED", "LOITER", "RTL", "LAND", "GUIDED_NOGPS"]
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
            "alt": self._get_altitude() or 0.0,
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
            if not self._is_mcu_serial_port(msg.port):
                return

            # 解析串口数据
            data = bytes(msg.data[:msg.count])
            for resp_bytes, resp_str in self._response_map.items():
                if resp_bytes in data:
                    self._response_buffer = resp_str
                    logger.debug(f"收到 Pico 2 响应：{resp_str}")
                    break

        return _on_serial_control

    def _serial_control_port(self) -> int:
        """Map a logical UART number to the MAVLink SERIAL_CONTROL enum."""
        port = self.config.mcu_serial_port
        if port >= 100:
            return port
        constant_name = f"SERIAL_CONTROL_SERIAL{port}"
        return getattr(mavutil.mavlink, constant_name, port)

    def _is_mcu_serial_port(self, port: int) -> bool:
        return port in {self.config.mcu_serial_port, self._serial_control_port()}

    def connect(self) -> bool:
        """Validate the legacy Pixhawk SERIAL_CONTROL MCU path."""
        if not self._flight_bridge.is_connected():
            logger.error("Cannot connect MCU through Pixhawk: flight link is closed")
            return False
        return self._register_serial_listener()

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

        # 注册 SERIAL_CONTROL 监听
        self._vehicle.add_message_listener('SERIAL_CONTROL', self._serial_listener)
        logger.info(f"[MCUBridge] 已注册 SERIAL_CONTROL 监听")

        # 同时注册调试用的 ALL 监听器（捕获 SERIAL_CONTROL 消息）
        def _debug_all_messages(vehicle, name, msg):
            if name == 'SERIAL_CONTROL':
                logger.debug(f"[MCUBridge-DEBUG] 捕获到 SERIAL_CONTROL 消息")


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
            data_array = bytearray(70)
            data_array[:len(cmd_bytes)] = cmd_bytes
            msg = self._vehicle.message_factory.serial_control_encode(
                self._serial_control_port(),  # 目标串口
                0,  # 预留
                0,  # 操作：写入
                self.config.mcu_baudrate,  # 波特率
                len(cmd_bytes),  # 数据长度
                bytes(data_array)  # 指令数据（70 字节）
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
