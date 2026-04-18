"""
Phase 2 桌面联调 Mock 替身。
"""

import cv2
import glob
import time
import logging
import threading
from pathlib import Path

import numpy as np

from core.interfaces import (
    IFlightBridge, IMCUBridge, IStreamer,
    FlightMode, MCUCommand, MCUResponse,
)
from utils.logger import setup_logger


class MockStreamer(IStreamer):
    """
    从本地视频/图片序列模拟摄像头流。
    接口与 ZeroLatencyStreamer 一致，用于 Phase 2 桌面联调。

    支持两种数据源：
      1. 视频文件（如 "test_data/flight.mp4"）
      2. 图片目录（如 "test_data/frames/"），自动扫描 .jpg/.png
    """

    def __init__(self, source: str, loop: bool = True) -> None:
        """
        Args:
            source: 数据源路径（视频文件或图片目录）
            loop:   是否循环播放（默认 True，用于持续联调）

        Raises:
            FileNotFoundError: 数据源路径不存在
        """
        self._source = source
        self._loop = loop
        self._logger = setup_logger("Mock.Streamer", level="INFO")

        source_path = Path(source)

        if source_path.is_file():
            # ── 视频文件模式 ──
            self._is_video = True
            self._cap = cv2.VideoCapture(str(source_path))
            self._frames: list[str] = []
            self._frame_idx = 0
            self._opened = self._cap.isOpened()
            if self._opened:
                self._logger.info(f"MockStreamer: 视频模式 → {source}")
            else:
                self._logger.warning(f"MockStreamer: 无法打开视频文件 → {source}")

        elif source_path.is_dir():
            # ── 图片目录模式 ──
            self._is_video = False
            self._cap = None
            self._frames = sorted(
                glob.glob(str(source_path / "*.jpg"))
                + glob.glob(str(source_path / "*.png"))
            )
            self._frame_idx = 0
            self._opened = len(self._frames) > 0
            if self._opened:
                self._logger.info(
                    f"MockStreamer: 图片模式 → {source} ({len(self._frames)} 帧)"
                )
            else:
                self._logger.warning(f"MockStreamer: 图片目录为空 → {source}")

        else:
            raise FileNotFoundError(f"数据源不存在: {source}")

    def get_latest_frame(self) -> np.ndarray | None:
        """
        Returns:
            BGR 图像帧（H×W×3, np.uint8），或 None
        """
        if not self._opened:
            return None

        if self._is_video:
            return self._read_video_frame()
        else:
            return self._read_image_frame()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._opened = False

    def is_opened(self) -> bool:
        return self._opened

    def _read_video_frame(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return None
            else:
                self._opened = False
                return None
        return frame

    def _read_image_frame(self) -> np.ndarray | None:
        if self._frame_idx >= len(self._frames):
            if self._loop:
                self._frame_idx = 0
            else:
                self._opened = False
                return None
        frame = cv2.imread(self._frames[self._frame_idx])
        self._frame_idx += 1
        return frame


class MockFlightBridge(IFlightBridge):
    """
    飞控替身 — 模拟飞行行为，记录所有收到的指令。

      - _alt: 模拟高度（起飞时设置，send_body_velocity 时根据 vz 递变）
      - _armed: 解锁状态
      - _mode: 飞行模式
      - _velocity_log: 所有 send_body_velocity 调用的记录列表
    """

    # dt 估计值，与 tick_rate_hz=15 对应
    _DT_ESTIMATE = 0.066

    def __init__(self) -> None:
        # ── 模拟飞行状态 ──
        self._connected: bool = False
        self._armed: bool = False
        self._mode: str = FlightMode.GUIDED
        self._alt: float = 0.0
        self._heading: float = 0.0
        self._battery_pct: float = 1.0
        self._heartbeat_ok: bool = True

        # ── 可编程返回值（测试用） ──
        self._takeoff_return: bool = True
        self._land_return: bool = True

        # ── 指令记录 ──
        self._velocity_log: list[dict] = []
        self._command_log: list[dict] = []

        # ── 日志 ──
        self._logger = setup_logger("Mock.Flight", level="INFO")

    def connect(self) -> bool:
        """模拟连接：始终成功。"""
        self._connected = True
        self._heartbeat_ok = True
        self._command_log.append({"time": time.time(), "action": "connect"})
        self._logger.info("MockFlight: 连接成功")
        return True

    def arm_and_takeoff(self, target_alt: float) -> bool:
        """模拟起飞：armed=True, alt=target_alt。"""
        self._command_log.append({
            "time": time.time(),
            "action": "arm_and_takeoff",
            "target_alt": target_alt,
        })
        if not self._takeoff_return:
            self._logger.warning("MockFlight: 起飞失败（编程返回 False）")
            return False
        self._armed = True
        self._mode = FlightMode.GUIDED
        self._alt = target_alt
        self._logger.info(f"MockFlight: 起飞完成 → alt={target_alt:.2f}")
        return True

    def send_body_velocity(
        self, vx: float, vy: float, vz: float, yaw_rate: float
    ) -> None:
        """
        NED 坐标系：vz > 0 表示向下，因此 alt -= vz * dt。
        """
        self._velocity_log.append({
            "time": time.time(),
            "vx": vx, "vy": vy, "vz": vz, "yaw_rate": yaw_rate,
        })

        self._alt = max(0.0, self._alt - vz * self._DT_ESTIMATE)

        self._logger.debug(
            f"MockFlight: vel({vx:.3f}, {vy:.3f}, {vz:.3f}, {yaw_rate:.3f}) "
            f"→ alt={self._alt:.3f}"
        )

    def land(self) -> bool:
        """模拟降落：alt=0, armed=False。"""
        self._command_log.append({"time": time.time(), "action": "land"})
        if not self._land_return:
            self._logger.warning("MockFlight: 降落失败（编程返回 False）")
            return False
        self._mode = FlightMode.LAND
        self._alt = 0.0
        self._armed = False
        self._logger.info("MockFlight: 降落完成")
        return True

    def set_mode(self, mode: str) -> bool:
        self._command_log.append({
            "time": time.time(), "action": "set_mode", "mode": mode,
        })
        self._mode = mode
        self._logger.info(f"MockFlight: 模式切换 → {mode}")
        return True

    def get_telemetry(self) -> dict:
        return {
            "armed": self._armed,
            "mode": self._mode,
            "alt": self._alt,
            "heading": self._heading,
            "battery_pct": self._battery_pct,
            "heartbeat_ok": self._heartbeat_ok,
        }

    def is_connected(self) -> bool:
        return self._connected

    def get_command_log(self) -> list[dict]:
        return list(self._command_log)

    def get_velocity_log(self) -> list[dict]:
        return list(self._velocity_log)

    def set_heartbeat_ok(self, ok: bool) -> None:
        self._heartbeat_ok = ok

    def set_connected(self, connected: bool) -> None:
        self._connected = connected


# ═══════════════════════════════════════════════════════
#  MockMCUBridge — MCU 通信模拟
# ═══════════════════════════════════════════════════════

class MockMCUBridge(IMCUBridge):
    """
    模拟舵机/红外反馈。

    用法示例：
        mock_mcu = MockMCUBridge()
        mock_mcu.set_auto_response("START_GRAB", "GRAB_DONE", delay_s=2.0)
        mock_mcu.inject_failure("START_GRAB", "GRAB_FAIL")
    """

    def __init__(self) -> None:
        """初始化 MCU 模拟，预设默认的正常响应。"""
        # ── 自动响应规则（持久生效） ──
        self._auto_responses: dict[str, tuple[str, float]] = {}

        # ── 故障注入队列（一次性消耗） ──
        self._failure_queue: dict[str, str] = {}

        # ── 响应缓冲区 ──
        self._pending_response: str | None = None

        # ── 定时器管理 ──
        self._active_timer: threading.Timer | None = None

        # ── 指令记录 ──
        self._command_log: list[dict] = []

        # ── 连接状态 ──
        self._connected: bool = True

        # ── 日志 ──
        self._logger = setup_logger("Mock.MCU", level="INFO")

        # ── 预设默认正常响应 ──
        self.set_auto_response(
            MCUCommand.RESET, MCUResponse.RESET_DONE, delay_s=0.5
        )
        self.set_auto_response(
            MCUCommand.START_GRAB, MCUResponse.GRAB_DONE, delay_s=2.0
        )
        self.set_auto_response(
            MCUCommand.START_RELEASE, MCUResponse.RELEASE_DONE, delay_s=1.5
        )

    def send_command(self, command: str) -> bool:
        """
        优先级：故障注入 > 自动响应 > 无规则（仅记录）
        """
        self._command_log.append({
            "time": time.time(),
            "command": command,
        })

        # 取消之前的定时器（如有）
        if self._active_timer is not None:
            self._active_timer.cancel()
            self._active_timer = None

        # 清空待读取响应
        self._pending_response = None

        # 优先检查故障注入队列
        if command in self._failure_queue:
            fail_response = self._failure_queue.pop(command)
            delay_s = 0.0
            if command in self._auto_responses:
                _, delay_s = self._auto_responses[command]
            self._schedule_response(fail_response, delay_s)
            self._logger.info(
                f"MockMCU: 故障注入 {command} → {fail_response} "
                f"(delay={delay_s}s)"
            )
            return True

        # 正常自动响应
        if command in self._auto_responses:
            response, delay_s = self._auto_responses[command]
            self._schedule_response(response, delay_s)
            self._logger.info(
                f"MockMCU: 自动响应 {command} → {response} "
                f"(delay={delay_s}s)"
            )
            return True

        # 无规则：仅记录，不产生响应
        self._logger.warning(
            f"MockMCU: 收到未知指令 {command}，无响应规则"
        )
        return True

    def get_latest_response(self) -> str | None:
        """
        返回后清空缓冲区，下次调用返回 None（除非新响应已到达）。
        """
        resp = self._pending_response
        if resp is not None:
            self._pending_response = None
        return resp

    def is_connected(self) -> bool:
        return self._connected

    def set_auto_response(
        self, trigger_cmd: str, response: str, delay_s: float = 1.0
    ) -> None:
        """
        Args:
            trigger_cmd: 触发指令（如 MCUCommand.START_GRAB）
            response:    自动响应（如 MCUResponse.GRAB_DONE）
            delay_s:     延迟时间（秒，模拟真实 MCU 动作耗时）
        """
        self._auto_responses[trigger_cmd] = (response, delay_s)

    def inject_failure(
        self, trigger_cmd: str, failure_response: str
    ) -> None:
        """
        Args:
            trigger_cmd:      触发指令
            failure_response: 故障响应（如 MCUResponse.GRAB_FAIL）
        """
        self._failure_queue[trigger_cmd] = failure_response

    def get_command_log(self) -> list[dict]:
        return list(self._command_log)

    def set_connected(self, connected: bool) -> None:
        self._connected = connected

    def clear_auto_responses(self) -> None:
        self._auto_responses.clear()

    def _schedule_response(self, response: str, delay_s: float) -> None:
        def _deliver():
            self._pending_response = response
            self._logger.debug(f"MockMCU: 响应就绪 → {response}")

        if delay_s <= 0:
            _deliver()
        else:
            self._active_timer = threading.Timer(delay_s, _deliver)
            self._active_timer.daemon = True
            self._active_timer.start()
