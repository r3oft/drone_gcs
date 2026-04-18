"""
本模块定义 IFlightBridge 和 IMCUBridge 两个抽象基类，
以及相关的类型定义和协议常量。

实现方：
  - 真实实现：飞控组 core/flight_bridge.py（FlightBridge, MCUBridge）
  - Mock 实现：算法组 utils/mock.py（MockFlightBridge, MockMCUBridge）

消费方：
  - 算法组 core/state_machine.py（GlobalFSM）
"""

from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np

class TelemetryData(TypedDict):
    """get_telemetry() 返回的遥测数据结构。"""
    armed: bool            # 是否已解锁
    mode: str              # 当前飞行模式 (e.g. "GUIDED")
    alt: float             # 相对高度 (m)
    heading: float         # 航向角 (deg, 0-360)
    battery_pct: float     # 电池剩余百分比 (0.0~1.0)
    heartbeat_ok: bool     # 心跳链路是否正常

class MCUCommand:
    """MCU 下行指令常量（PC → Pico 2）。"""
    START_GRAB    = "START_GRAB"
    START_RELEASE = "START_RELEASE"
    RESET         = "RESET"

    ALL = frozenset({START_GRAB, START_RELEASE, RESET})


class MCUResponse:
    """MCU 上行响应常量（Pico 2 → PC）。"""
    GRAB_DONE    = "GRAB_DONE"
    GRAB_FAIL    = "GRAB_FAIL"
    RELEASE_DONE = "RELEASE_DONE"
    RELEASE_FAIL = "RELEASE_FAIL"
    RESET_DONE   = "RESET_DONE"

    ALL = frozenset({GRAB_DONE, GRAB_FAIL, RELEASE_DONE, RELEASE_FAIL, RESET_DONE})


class FlightMode:
    """飞行模式常量。"""
    GUIDED = "GUIDED"
    LOITER = "LOITER"
    RTL    = "RTL"
    LAND   = "LAND"

    ALL = frozenset({GUIDED, LOITER, RTL, LAND})


class IFlightBridge(ABC):
    """
    飞行控制抽象接口。

    由飞控组在 core/flight_bridge.py 中实现真实版本，
    由算法组在 utils/mock.py 中实现 Mock 版本。
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        建立与飞控器的 MAVLink 连接，阻塞等待首个心跳包。

        Returns:
            True: 连接成功
            False: 连接超时或失败
        """
        ...

    @abstractmethod
    def arm_and_takeoff(self, target_alt: float) -> bool:
        """
        解锁电机 + 起飞到指定相对高度。

        阻塞执行，内部循环检测高度直到满足以下任一条件：
          1. 当前高度 >= target_alt * 0.95 → 返回 True
          2. 等待超时（由 config flight.takeoff_timeout_s 控制）→ 返回 False

        Args:
            target_alt: 目标高度（米），相对于起飞点

        Returns:
            True: 成功到达目标高度
            False: 超时未到达
        """
        ...

    @abstractmethod
    def send_body_velocity(
        self, vx: float, vy: float, vz: float, yaw_rate: float
    ) -> None:
        """
        发送机体坐标系速度指令。

        坐标系：MAV_FRAME_BODY_NED
            vx:       前向速度 (m/s)，正值 = 向前
            vy:       侧向速度 (m/s)，正值 = 向右
            vz:       垂直速度 (m/s)，正值 = 向下（NED 约定）
            yaw_rate: 偏航角速度 (rad/s)，正值 = 顺时针

        非阻塞，立即返回。飞控器收到后持续执行约 1 秒，
        因此需要以 ≥1Hz 的频率持续发送以保持控制。

        注意：对应 MAVLink #84 SET_POSITION_TARGET_LOCAL_NED，
             type_mask=0x07C7（纯速度接管）。
        """
        ...

    @abstractmethod
    def land(self) -> bool:
        """
        切换到 LAND 模式并等待触地。

        阻塞执行，内部循环检测直到满足以下任一条件：
          1. 飞控解除 armed 状态 → 返回 True
          2. 相对高度 < land_detect_alt → 返回 True
          3. 等待超时 → 返回 False

        Returns:
            True: 已触地
            False: 超时未触地
        """
        ...

    @abstractmethod
    def set_mode(self, mode: str) -> bool:
        """
        切换飞行模式。

        Args:
            mode: 目标模式字符串，取值范围：
                  "GUIDED" — 引导模式（接受外部速度指令）
                  "LOITER" — 定点悬停
                  "RTL"    — 自动返航

        Returns:
            True: 模式切换成功
            False: 切换失败
        """
        ...

    @abstractmethod
    def get_telemetry(self) -> TelemetryData:
        """
        获取当前飞行遥测数据（非阻塞）。

        Returns:
            TelemetryData 字典，包含以下字段：
            {
                "armed": bool,           # 是否已解锁
                "mode": str,             # 当前飞行模式 (e.g. "GUIDED")
                "alt": float,            # 相对高度 (m)
                "heading": float,        # 航向角 (deg, 0-360)
                "battery_pct": float,    # 电池剩余百分比 (0.0~1.0)
                "heartbeat_ok": bool,    # 心跳链路是否正常
            }
        """
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """检查飞控连接是否存活（心跳未超时）。"""
        ...


class IMCUBridge(ABC):
    """
    末端执行器（Pico 2）通信抽象接口。

    由飞控组在 core/mcu_bridge.py 中实现真实版本，
    由算法组在 utils/mock.py 中实现 Mock 版本。

    通信路径：PC ←(MAVLink SERIAL_CONTROL)→ Pixhawk ←(UART4)→ Pico 2
    """

    @abstractmethod
    def send_command(self, command: str) -> bool:
        """
        向 Pico 2 发送控制指令。

        Args:
            command: 指令字符串，取值范围：
                "START_GRAB"    — 执行抓取（舵机闭合 + 红外检测）
                "START_RELEASE" — 执行释放（舵机张开 + 红外检测）
                "RESET"         — 舵机复位到初始位置

        Returns:
            True: 指令成功发送（不代表动作完成，完成需通过 get_latest_response 确认）
            False: 发送失败（链路断开等）
        """
        ...

    @abstractmethod
    def get_latest_response(self) -> str | None:
        """
        非阻塞读取 Pico 2 的最新响应。

        Returns:
            响应字符串，可能的取值：
                "GRAB_DONE"     — 抓取完成（红外确认货物已装载）
                "GRAB_FAIL"     — 抓取失败（红外确认货物未装载）
                "RELEASE_DONE"  — 释放完成（红外确认货物已脱离）
                "RELEASE_FAIL"  — 释放失败（红外确认货物未脱离）
                "RESET_DONE"    — 复位完成
            None: 暂无新响应
        """
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """检查 MCU 通信链路是否存活。"""
        ...


class IStreamer(ABC):
    """
    视频流采集抽象接口。

    真实实现：core/streamer.py（ZeroLatencyStreamer, Phase 3）
    Mock 实现：utils/mock.py（MockStreamer, Phase 2）
    """

    @abstractmethod
    def get_latest_frame(self) -> np.ndarray | None:
        """
        获取当前最新帧（非阻塞）。

        Returns:
            BGR 图像帧（H×W×3, np.uint8），或 None（流未就绪/断连）
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """释放底层资源（VideoCapture / 网络套接字等）。"""
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        """检查流是否处于可用状态。"""
        ...