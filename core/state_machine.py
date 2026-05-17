"""
全局有限状态机（M5 模块）。
"""

from __future__ import annotations

import time
import logging
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from core.interfaces import (
    IFlightBridge, IMCUBridge,
    MCUCommand, MCUResponse, FlightMode,
)
from utils.config_manager import ConfigManager
from utils.logger import setup_logger, FlightRecorder

if TYPE_CHECKING:
    from core.perception import TargetPoseEstimator
    from core.servo_controller import VisualServoController


MISSION_STANDARD = "standard"
MISSION_NO_MCU_FLIGHT_TEST = "no_mcu_flight_test"


class FlightState(Enum):
    """
    状态枚举。
        0-9   系统阶段
        10-19 取货流程
        20-29 转运/投递流程
        30-39 返航/循环
        90+   异常/安全
    """
    # ─── 系统阶段 ───
    IDLE                = 0     # 系统空闲，等待启动指令
    RESET               = 1     # 系统自检 + 舵机复位

    # ─── 取货流程 ───
    INBOUND             = 10    # 解锁起飞 + 飞往取货区上方
    TASK_REC_ALIGN      = 11    # 视觉伺服对准取货区 (target_cls_id=0, pickup_zone)
    TASK_REC_DESCEND    = 12    # 盲降到取货区（维持平面纠偏 + 注入下降速度）
    TASK_REC_WAIT_LOAD  = 13    # 悬停等待人工装填 + 红外确认装填完成

    # ─── 转运 & 投递流程 ───
    TRANS_DELIVERY      = 20    # 二次起飞 + 定距转移飞往投递区（视觉/PD 不工作）
    TASK_REL_ALIGN      = 21    # 视觉伺服对准投递区 (target_cls_id=1, delivery_zone)
    TASK_REL_DESCEND    = 22    # 盲降到投递区
    TASK_REL_RELEASE    = 23    # 舵机释放货物 + 红外校验释放成功

    # ─── 返航/循环 ───
    TRANS_CARGO         = 30    # 起飞 + 飞往下一个取货区
    OUTBOUND            = 31    # 最终返航降落 + 停桨

    # ─── 异常/安全 ───
    EMERGENCY           = 90    # 紧急状态（Failsafe 触发后的统一兜底）


class GlobalFSM:
    """
    全局有限状态机（M5）。
    """

    _DESCEND_VZ = 0.15      # 盲降下降速度 (m/s)，NED 向下为正
    _DESCEND_GAIN = 0.5     # 盲降阶段平面纠偏增益衰减
    _RESET_TIMEOUT_S = 30.0 # RESET 状态超时 (s)

    _VISION_CLS_MAP = {
        FlightState.TASK_REC_ALIGN:   0,  # pickup_zone
        FlightState.TASK_REC_DESCEND: 0,
        FlightState.TASK_REL_ALIGN:   1,  # delivery_zone
        FlightState.TASK_REL_DESCEND: 1,
    }

    def __init__(
        self,
        flight_bridge: IFlightBridge,
        mcu_bridge: IMCUBridge,
        perception: TargetPoseEstimator,
        controller: VisualServoController,
        config: ConfigManager,
    ) -> None:
        """
        Args:
            flight_bridge: 飞行控制接口（真实 or Mock）
            mcu_bridge:    MCU 通信接口（真实 or Mock）
            perception:    M2 视觉推理模块（已完成）
            controller:    M3 视觉伺服控制器（已完成）
            config:        全局配置管理器（已完成）
        """
        self._flight = flight_bridge
        self._mcu = mcu_bridge
        self._perception = perception
        self._controller = controller

        self._mission_profile: str = str(config.get("mission.profile", MISSION_STANDARD))
        self._no_mcu_flight_test: bool = (
            self._mission_profile == MISSION_NO_MCU_FLIGHT_TEST
        )
        self._camera_link_loss_timeout: float = float(
            config.get("mission.camera_link_loss_timeout_s", 5.0)
        )
        self._max_flight_alt_m: float = float(
            config.get("mission.max_flight_alt_m", 0.5)
        )

        self._reset_timeout: float = float(
            config.get("fsm.reset_timeout_s", self._RESET_TIMEOUT_S)
        )
        self._align_hold_time: float = config.get("fsm.align_hold_time_s", 1.5)
        self._target_lost_hover: float = config.get("fsm.target_lost_hover_s", 1.0)
        self._target_lost_climb: float = config.get("fsm.target_lost_climb_s", 3.0)
        self._climb_vz: float = config.get("fsm.climb_vz", -0.2)
        self._takeoff_alt: float = config.get("flight.takeoff_alt", 1.5)
        self._takeoff_alt_tolerance: float = config.get(
            "flight.takeoff_alt_tolerance_m", 0.12
        )
        self._takeoff_timeout: float = config.get("flight.takeoff_timeout_s", 15.0)
        self._land_detect_alt: float = config.get("flight.land_detect_alt", 0.15)
        self._grab_timeout: float = config.get("mcu.grab_timeout_s", 10.0)
        self._release_timeout: float = config.get("mcu.release_timeout_s", 10.0)
        self._retry_max: int = config.get("mcu.retry_max", 2)
        self._transfer_speed: float = config.get("transfer.transfer_speed", 0.3)
        self._transfer_distance: float = config.get("transfer.delivery_distance_m", 3.0)
        self._transfer_alt: float = config.get("transfer.transfer_alt", 1.5)
        self._pre_pickup_forward_m: float = config.get("mission.pre_pickup_forward_m", 0.0)
        self._pre_pickup_speed_mps: float = config.get("mission.pre_pickup_speed_mps", 0.2)
        self._no_mcu_pickup_use_land_mode: bool = bool(
            config.get("mission.no_mcu_pickup_use_land_mode", False)
        )
        self._no_mcu_pickup_touchdown_alt_m: float = config.get(
            "mission.no_mcu_pickup_touchdown_alt_m", 0.08
        )
        self._no_mcu_pickup_touchdown_hold_s: float = config.get(
            "mission.no_mcu_pickup_touchdown_hold_s", 0.4
        )
        self._no_mcu_pickup_min_descend_s: float = config.get(
            "mission.no_mcu_pickup_min_descend_s", 0.8
        )
        self._no_mcu_pickup_descend_vz_mps: float = config.get(
            "mission.no_mcu_pickup_descend_vz_mps", 0.12
        )
        self._no_mcu_pickup_descend_timeout_s: float = config.get(
            "mission.no_mcu_pickup_descend_timeout_s", 8.0
        )
        self._no_mcu_retakeoff_climb_vz_mps: float = config.get(
            "mission.no_mcu_retakeoff_climb_vz_mps", 0.1
        )
        self._no_mcu_ignore_yaw_alignment: bool = bool(
            config.get("mission.no_mcu_ignore_yaw_alignment", False)
        )
        self._no_mcu_retakeoff_delay_s: float = config.get(
            "mission.no_mcu_retakeoff_delay_s", 2.0
        )
        self._no_mcu_retakeoff_wait_timeout_s: float = config.get(
            "mission.no_mcu_retakeoff_wait_timeout_s", 12.0
        )
        self._no_mcu_retakeoff_require_disarmed: bool = bool(
            config.get("mission.no_mcu_retakeoff_require_disarmed", True)
        )
        if self._max_flight_alt_m <= 0:
            raise ValueError("mission.max_flight_alt_m must be > 0")
        if self._takeoff_alt_tolerance < 0:
            raise ValueError("flight.takeoff_alt_tolerance_m must be >= 0")
        if self._pre_pickup_forward_m < 0:
            raise ValueError("mission.pre_pickup_forward_m must be >= 0")
        if self._pre_pickup_speed_mps <= 0:
            raise ValueError("mission.pre_pickup_speed_mps must be > 0")
        if self._no_mcu_pickup_touchdown_alt_m < 0:
            raise ValueError("mission.no_mcu_pickup_touchdown_alt_m must be >= 0")
        if self._no_mcu_pickup_touchdown_hold_s < 0:
            raise ValueError("mission.no_mcu_pickup_touchdown_hold_s must be >= 0")
        if self._no_mcu_pickup_min_descend_s < 0:
            raise ValueError("mission.no_mcu_pickup_min_descend_s must be >= 0")
        if self._no_mcu_pickup_descend_vz_mps <= 0:
            raise ValueError("mission.no_mcu_pickup_descend_vz_mps must be > 0")
        if self._no_mcu_pickup_descend_timeout_s <= 0:
            raise ValueError("mission.no_mcu_pickup_descend_timeout_s must be > 0")
        if self._no_mcu_retakeoff_climb_vz_mps <= 0:
            raise ValueError("mission.no_mcu_retakeoff_climb_vz_mps must be > 0")
        if self._no_mcu_retakeoff_delay_s < 0:
            raise ValueError("mission.no_mcu_retakeoff_delay_s must be >= 0")
        if self._no_mcu_retakeoff_wait_timeout_s <= 0:
            raise ValueError("mission.no_mcu_retakeoff_wait_timeout_s must be > 0")
        self._takeoff_alt = min(self._takeoff_alt, self._max_flight_alt_m)
        self._transfer_alt = min(self._transfer_alt, self._max_flight_alt_m)
        self._center_u: float = config.get("camera.center_u", 320)
        self._center_v: float = config.get("camera.center_v", 240)

        # ── FSM 核心状态 ──
        self._state: FlightState = FlightState.IDLE
        self._start_requested: bool = False
        self._stop_requested: bool = False

        # ── 防抖跃迁计时器 ──
        self._align_stable_start: float = 0.0

        # ── 目标丢失看门狗 ──
        self._last_target_seen: float = time.time()
        self._camera_link_loss_start: float = 0.0

        # ── MCU 交互状态 ──
        self._mcu_cmd_sent: bool = False
        self._mcu_cmd_time: float = 0.0
        self._mcu_retry_count: int = 0

        # ── 定距转移状态 ──
        self._transfer_start_time: float = 0.0
        self._transfer_takeoff_done: bool = False
        self._inbound_takeoff_done: bool = False
        self._inbound_transfer_start_time: float = 0.0
        self._post_land_wait_required: bool = False
        self._post_land_wait_start_time: float = 0.0
        self._post_land_wait_logged: bool = False
        self._no_mcu_pickup_descend_start_time: float = 0.0
        self._no_mcu_pickup_touchdown_start_time: float = 0.0
        self._no_mcu_guided_climb_logged: bool = False

        # ── 时间管理 ──
        self._last_tick_time: float = 0.0
        self._state_enter_time: float = time.time()

        # ── 黑匣子记录上下文（每个 tick 更新） ──
        self._tick_target: dict | None = None
        self._tick_vel: tuple[float, float, float] | None = None

        # ── 日志 ──
        log_level = config.get("logging.level", "INFO")
        log_dir = config.get("logging.log_dir", "logs/")
        self._logger = setup_logger("M5.FSM", level=log_level, log_dir=log_dir)

        enable_recorder = config.get("logging.enable_flight_recorder", True)
        if enable_recorder:
            self._recorder: FlightRecorder | None = FlightRecorder(log_dir=log_dir)
        else:
            self._recorder = None

    def tick(self, frame: np.ndarray | None) -> None:
        """
        主循环单步执行。

        由 main.py 的主循环每 tick 调用一次。
        根据当前状态执行对应逻辑：
          - 视觉阶段 → perception.process_frame() → controller.compute_velocity()
                     → flight_bridge.send_body_velocity()
          - 非视觉阶段 → mcu_bridge / flight_bridge 的对应操作

        Args:
            frame: 当前摄像头帧（BGR, H×W×3）。
                   非视觉阶段可传入 None。
        """
        now = time.time()
        dt = now - self._last_tick_time if self._last_tick_time > 0 else 0.0
        self._last_tick_time = now

        # 重置每 tick 的记录上下文
        self._tick_target = None
        self._tick_vel = None
        self._tick_debug = None

        # ── 层 0：全局 Failsafe 拦截 ──
        # IDLE/RESET 阶段跳过连接类 Failsafe（初始未连接是正常状态）
        if self._state not in (
            FlightState.IDLE, FlightState.RESET, FlightState.EMERGENCY
        ):
            self._check_failsafe()

        # ── 层 1：状态分派 ──
        handler = self._STATE_HANDLERS.get(self._state)
        if handler:
            handler(self, frame, dt, now)

        # ── 层 2：黑匣子记录 ──
        if self._recorder:
            target = self._tick_target
            vel = self._tick_vel
            self._recorder.record({
                "state":        self._state.name,
                "target_u":     target["u"] if target else "",
                "target_v":     target["v"] if target else "",
                "target_theta": target["theta"] if target else "",
                "target_conf":  target["conf"] if target else "",
                "cmd_vx":       vel[0] if vel else "",
                "cmd_vy":       vel[1] if vel else "",
                "cmd_vyaw":     vel[2] if vel else "",
                "dt":           dt,
            })

    @property
    def state(self) -> FlightState:
        """获取当前 FSM 状态。"""
        return self._state

    def request_start(self) -> None:
        """外部触发：请求从 IDLE 启动任务。"""
        if self._state == FlightState.IDLE:
            self._start_requested = True
            self._logger.info("收到启动请求")

    def request_stop(self) -> None:
        """外部触发：请求中止任务并进入 OUTBOUND 返航。"""
        if self._state not in (
            FlightState.IDLE, FlightState.EMERGENCY, FlightState.OUTBOUND
        ):
            self._stop_requested = True
            self._logger.info("收到停止请求，准备返航")
            self._transition_to(FlightState.OUTBOUND)


    def _transition_to(self, new_state: FlightState) -> None:
        """状态跃迁统一入口：写日志、重置计时器和阶段性标志。"""
        old_state = self._state
        self._state = new_state
        now = time.time()
        self._state_enter_time = now

        # 重置所有阶段性状态
        self._align_stable_start = 0.0
        self._mcu_cmd_sent = False
        self._mcu_retry_count = 0
        self._transfer_takeoff_done = False
        self._inbound_takeoff_done = False
        self._inbound_transfer_start_time = 0.0
        self._post_land_wait_required = False
        self._post_land_wait_start_time = 0.0
        self._post_land_wait_logged = False
        self._no_mcu_pickup_descend_start_time = 0.0
        self._no_mcu_pickup_touchdown_start_time = 0.0
        self._no_mcu_guided_climb_logged = False
        self._camera_link_loss_start = 0.0

        # 进入视觉状态时初始化目标看门狗，
        # 防止首帧 target==None 时误判为长时间丢失而触发爬升
        if new_state in self._VISION_CLS_MAP:
            self._last_target_seen = now

        self._logger.info(f"状态跃迁：{old_state.name} → {new_state.name}")

    def _get_target_cls_id(self) -> int:
        """根据当前状态返回目标类别 ID，非视觉阶段返回 -1。"""
        return self._VISION_CLS_MAP.get(self._state, -1)

    def _check_failsafe(self) -> None:
        """检测飞控层面异常，必要时跃迁至 EMERGENCY。"""
        if not self._flight.is_connected():
            self._logger.error("飞控连接断开，进入 EMERGENCY")
            self._transition_to(FlightState.EMERGENCY)
            return

        tel = self._flight.get_telemetry()
        if not tel["heartbeat_ok"]:
            self._logger.error("飞控心跳丢失，进入 EMERGENCY")
            self._transition_to(FlightState.EMERGENCY)
            return

        # 飞控被强制切换到 RTL / LAND（RC 接管或 Failsafe 触发）
        if tel["mode"] in (FlightMode.RTL, FlightMode.LAND):
            landed_between_no_mcu_legs = (
                self._no_mcu_flight_test
                and self._state == FlightState.TRANS_DELIVERY
                and not self._transfer_takeoff_done
                and (not tel["armed"] or tel["alt"] < self._land_detect_alt)
            )
            if self._state not in (FlightState.OUTBOUND, FlightState.EMERGENCY):
                if landed_between_no_mcu_legs:
                    return
                self._logger.warning(
                    f"飞控模式被外部切换为 {tel['mode']}，进入 EMERGENCY"
                )
                self._transition_to(FlightState.EMERGENCY)

    def _land_and_transition(self, success_state: FlightState, reason: str) -> None:
        self._controller.reset()
        if self._flight.land():
            self._logger.info(f"{reason}: landing complete")
            self._transition_to(success_state)
            if self._no_mcu_flight_test and success_state == FlightState.TRANS_DELIVERY:
                self._post_land_wait_required = True
                self._post_land_wait_start_time = time.time()
                self._post_land_wait_logged = False
        else:
            self._logger.error(f"{reason}: landing failed, entering EMERGENCY")
            self._transition_to(FlightState.EMERGENCY)

    def _wait_no_mcu_post_land_ready(self, now: float) -> bool:
        """
        Return True while TRANS_DELIVERY should keep waiting before second takeoff.

        FlightBridge.land() can report success once altitude is below the landing
        threshold, while ArduPilot is still in LAND and still armed.  Retakeoff at
        that instant can be ignored by the flight controller, so the no-MCU test
        waits for a stable landed state before calling arm_and_takeoff again.
        """
        if not self._post_land_wait_required:
            return False

        tel = self._flight.get_telemetry()
        elapsed = now - self._post_land_wait_start_time
        armed = bool(tel.get("armed", True))
        mode = tel.get("mode")
        alt = float(tel.get("alt", 0.0))
        delay_done = elapsed >= self._no_mcu_retakeoff_delay_s
        disarm_done = (not armed) or (not self._no_mcu_retakeoff_require_disarmed)

        if not self._post_land_wait_logged:
            self._post_land_wait_logged = True
            self._logger.info(
                "No-MCU retakeoff gate: waiting after pickup landing "
                f"(require_disarmed={self._no_mcu_retakeoff_require_disarmed}, "
                f"delay={self._no_mcu_retakeoff_delay_s:.1f}s)"
            )

        if delay_done and disarm_done:
            self._post_land_wait_required = False
            self._logger.info(
                "No-MCU retakeoff gate complete: "
                f"elapsed={elapsed:.1f}s mode={mode} armed={armed} alt={alt:.2f}m"
            )
            return False

        if elapsed >= self._no_mcu_retakeoff_wait_timeout_s:
            self._logger.error(
                "No-MCU retakeoff gate timeout: "
                f"elapsed={elapsed:.1f}s mode={mode} armed={armed} alt={alt:.2f}m; "
                "aborting mission on ground"
            )
            self._transition_to(FlightState.IDLE)
            return True

        return True

    def _get_current_altitude(self) -> tuple[float, dict]:
        tel = self._flight.get_telemetry()
        alt = float(tel.get("alt", 0.0) or 0.0)
        get_local_altitude = getattr(self._flight, "_get_local_altitude", None)
        if callable(get_local_altitude):
            try:
                local_alt = get_local_altitude()
                if local_alt is not None:
                    alt = float(local_alt)
                    tel["local_alt"] = alt
            except Exception:
                pass
        return alt, tel

    def _no_mcu_pickup_touchdown_reached(self, now: float) -> bool:
        alt, tel = self._get_current_altitude()
        elapsed = now - self._no_mcu_pickup_descend_start_time
        if elapsed < self._no_mcu_pickup_min_descend_s:
            self._no_mcu_pickup_touchdown_start_time = 0.0
            return False

        armed = bool(tel.get("armed", True))
        below_threshold = alt <= self._no_mcu_pickup_touchdown_alt_m
        if not below_threshold and armed:
            self._no_mcu_pickup_touchdown_start_time = 0.0
            return False

        if self._no_mcu_pickup_touchdown_start_time == 0.0:
            self._no_mcu_pickup_touchdown_start_time = now
            self._logger.info(
                "No-MCU pickup near-ground detected: "
                f"alt={alt:.2f}m threshold={self._no_mcu_pickup_touchdown_alt_m:.2f}m "
                f"armed={armed}"
            )
            return False

        return (
            now - self._no_mcu_pickup_touchdown_start_time
            >= self._no_mcu_pickup_touchdown_hold_s
        )

    def _handle_camera_link_loss(self, now: float) -> None:
        if self._camera_link_loss_start == 0.0:
            self._camera_link_loss_start = now
            self._logger.warning("Camera frame lost; hovering")

        self._flight.send_body_velocity(0, 0, 0, 0)
        self._tick_vel = (0, 0, 0)

        elapsed = now - self._camera_link_loss_start
        if elapsed >= self._camera_link_loss_timeout:
            self._logger.error(
                f"Camera link lost for {elapsed:.1f}s; landing and aborting mission"
            )
            self._land_and_transition(FlightState.IDLE, "Camera link failsafe")

    def _run_vision_pipeline(
        self, frame: np.ndarray | None, dt: float, now: float
    ) -> tuple[float, float, float] | None:
        """
        执行视觉感知 + 看门狗检查 + 伺服控制律。

        Returns:
            (vx, vy, vyaw) 速度输出，或 None（目标丢失，已由内部处理悬停/爬升）
        """
        cls_id = self._get_target_cls_id()
        if frame is None:
            self._handle_camera_link_loss(now)
            return None

        if self._camera_link_loss_start > 0.0:
            self._logger.info("Camera frame recovered; resuming visual control")
            self._camera_link_loss_start = 0.0

        target = self._perception.process_frame(frame, cls_id)

        if target is None:
            elapsed = now - self._last_target_seen
            if elapsed > self._target_lost_climb:
                # 二级干预：爬升搜索
                self._flight.send_body_velocity(0, 0, self._climb_vz, 0)
            elif elapsed > self._target_lost_hover:
                # 一级干预：悬停等待
                self._flight.send_body_velocity(0, 0, 0, 0)
            # 短于 hover 阈值：保持上一帧速度自然衰减
            return None

        self._last_target_seen = now
        self._tick_target = target
        compute_debug = getattr(self._controller, "compute_debug", None)
        if callable(compute_debug):
            debug = compute_debug(target, self._center_u, self._center_v, dt)
            self._tick_debug = debug
            vx, vy, vyaw = debug["velocities"]
        else:
            vx, vy, vyaw = self._controller.compute_velocity(
                target, self._center_u, self._center_v, dt
            )
        return (vx, vy, vyaw)

    def _apply_alignment_policy(
        self, vel: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if self._no_mcu_flight_test and self._no_mcu_ignore_yaw_alignment:
            vx, vy, _ = vel
            return (vx, vy, 0.0)
        return vel

    def _alignment_is_stable(self, vel: tuple[float, float, float]) -> bool:
        vx, vy, vyaw = vel
        if self._no_mcu_flight_test and self._no_mcu_ignore_yaw_alignment:
            return vx == 0.0 and vy == 0.0
        return vx == 0.0 and vy == 0.0 and vyaw == 0.0

    def _handle_mcu_action(
        self,
        command: str,
        success_response: str,
        fail_response: str,
        timeout: float,
        success_state: FlightState,
        now: float,
    ) -> None:
        """通用 MCU 指令-响应-重试逻辑。"""
        # 首次进入：发送指令
        if not self._mcu_cmd_sent:
            if not self._mcu.send_command(command):
                self._logger.error(f"MCU 指令发送失败: {command}")
                self._transition_to(FlightState.EMERGENCY)
                return
            self._mcu_cmd_sent = True
            self._mcu_cmd_time = time.time()
            self._mcu_retry_count = 0
            return

        # 检查响应
        resp = self._mcu.get_latest_response()
        if resp == success_response:
            self._transition_to(success_state)
            return

        # 失败或超时 → 重试
        if resp == fail_response or (now - self._mcu_cmd_time > timeout):
            if self._mcu_retry_count < self._retry_max:
                if not self._mcu.send_command(command):
                    self._logger.error(f"MCU 重试发送失败: {command}")
                    self._transition_to(FlightState.EMERGENCY)
                    return
                self._mcu_retry_count += 1
                self._mcu_cmd_time = now
                self._logger.warning(
                    f"MCU 重试 ({self._mcu_retry_count}/{self._retry_max}): {command}"
                )
            else:
                self._logger.error(
                    f"MCU 操作失败，已达最大重试次数: {command}"
                )
                self._transition_to(FlightState.EMERGENCY)

    def _takeoff_reached_threshold(self, target_alt: float) -> float:
        return max(
            0.05,
            min(target_alt * 0.95, target_alt - self._takeoff_alt_tolerance),
        )

    def _takeoff_altitude_reached(self, target_alt: float) -> bool:
        alt, _ = self._get_current_altitude()
        return alt >= self._takeoff_reached_threshold(target_alt)

    def _check_takeoff_timeout(
        self, now: float, target_alt: float, next_state: FlightState
    ) -> None:
        current_alt, tel = self._get_current_altitude()
        threshold = self._takeoff_reached_threshold(target_alt)
        if current_alt >= threshold:
            self._transition_to(next_state)
            return

        if now - self._mcu_cmd_time > self._takeoff_timeout:
            self._logger.error(
                f"起飞超时：{self._takeoff_timeout:.1f}s 未到达目标高度 "
                f"{target_alt:.2f}m（阈值 {threshold:.2f}m），当前 alt={current_alt:.2f}m "
                f"mode={tel.get('mode')} armed={tel.get('armed')}"
            )
            self._transition_to(FlightState.EMERGENCY)

    def _handle_idle(self, frame, dt: float, now: float) -> None:
        """IDLE：等待启动指令。"""
        if self._start_requested:
            self._start_requested = False
            self._transition_to(FlightState.RESET)

    def _handle_reset(self, frame, dt: float, now: float) -> None:
        """RESET：飞控连接 + MCU 复位 + 等待两者就绪。"""
        # 首次进入：执行连接和复位
        if not self._mcu_cmd_sent:
            if not self._flight.connect():
                self._logger.error("Flight connect failed, entering EMERGENCY")
                self._transition_to(FlightState.EMERGENCY)
                return
            if not self._mcu.connect():
                self._logger.error("MCU connect failed, entering EMERGENCY")
                self._transition_to(FlightState.EMERGENCY)
                return
            if not self._mcu.send_command(MCUCommand.RESET):
                self._logger.error("MCU RESET send failed, entering EMERGENCY")
                self._transition_to(FlightState.EMERGENCY)
                return
            self._mcu_cmd_sent = True
            self._mcu_cmd_time = time.time()
            return

        # 超时保护
        if now - self._mcu_cmd_time > self._reset_timeout:
            self._logger.error("RESET 状态超时，进入 EMERGENCY")
            self._transition_to(FlightState.EMERGENCY)
            return

        # 检查飞控心跳
        tel = self._flight.get_telemetry()
        heartbeat_ok = tel["heartbeat_ok"]

        # 检查 MCU 复位完成
        resp = self._mcu.get_latest_response()
        reset_done = (resp == MCUResponse.RESET_DONE)

        if heartbeat_ok and reset_done:
            self._transition_to(FlightState.INBOUND)

    def _handle_inbound(self, frame, dt: float, now: float) -> None:
        """INBOUND：解锁起飞到任务高度。"""
        if not self._mcu_cmd_sent:
            self._mcu_cmd_sent = True  # 复用标志防止重复调用
            success = self._flight.arm_and_takeoff(self._takeoff_alt)
            now = time.time()
            self._mcu_cmd_time = now
            if success is False:
                self._logger.error("起飞命令失败，进入 EMERGENCY")
                self._transition_to(FlightState.EMERGENCY)
                return

        if not self._inbound_takeoff_done:
            current_alt, tel = self._get_current_altitude()
            threshold = self._takeoff_reached_threshold(self._takeoff_alt)
            if current_alt >= threshold:
                self._inbound_takeoff_done = True
                self._logger.info(
                    f"起飞高度确认：alt={current_alt:.2f}m threshold={threshold:.2f}m"
                )
            elif now - self._mcu_cmd_time > self._takeoff_timeout:
                self._logger.error(
                    f"起飞超时：{self._takeoff_timeout:.1f}s 未到达目标高度 "
                    f"{self._takeoff_alt:.2f}m（阈值 {threshold:.2f}m），当前 alt={current_alt:.2f}m "
                    f"mode={tel.get('mode')} armed={tel.get('armed')}"
                )
                self._transition_to(FlightState.EMERGENCY)
                return
            else:
                return

        if self._pre_pickup_forward_m > 0:
            if self._inbound_transfer_start_time == 0.0:
                self._inbound_transfer_start_time = now
                self._logger.info(
                    f"起飞后前向定距转移开始：distance={self._pre_pickup_forward_m:.2f}m "
                    f"speed={self._pre_pickup_speed_mps:.2f}m/s"
                )

            expected_time = self._pre_pickup_forward_m / self._pre_pickup_speed_mps
            if now - self._inbound_transfer_start_time < expected_time:
                self._flight.send_body_velocity(self._pre_pickup_speed_mps, 0, 0, 0)
                self._tick_vel = (self._pre_pickup_speed_mps, 0, 0)
                return

            self._flight.send_body_velocity(0, 0, 0, 0)
            self._logger.info("起飞后前向定距转移完成")

        self._transition_to(FlightState.TASK_REC_ALIGN)

    def _handle_task_rec_align(self, frame, dt: float, now: float) -> None:
        """TASK_REC_ALIGN：视觉伺服对准取货区（cls_id=0）。"""
        vel = self._run_vision_pipeline(frame, dt, now)
        if vel is None:
            # 目标丢失，看门狗已在内部处理
            self._align_stable_start = 0.0
            return

        vel = self._apply_alignment_policy(vel)
        vx, vy, vyaw = vel
        self._tick_vel = vel

        # 下发速度指令（平面伺服，Vz=0）
        self._flight.send_body_velocity(vx, vy, 0.0, vyaw)

        # 防抖跃迁判定
        if self._alignment_is_stable(vel):
            if self._align_stable_start == 0.0:
                self._align_stable_start = now
            elif now - self._align_stable_start >= self._align_hold_time:
                self._controller.reset()
                if self._no_mcu_flight_test:
                    if self._no_mcu_pickup_use_land_mode:
                        self._land_and_transition(
                            FlightState.TRANS_DELIVERY,
                            "No-MCU pickup alignment",
                        )
                    else:
                        self._logger.info(
                            "No-MCU pickup alignment complete; starting guided descent"
                        )
                        self._transition_to(FlightState.TASK_REC_DESCEND)
                else:
                    self._transition_to(FlightState.TASK_REC_DESCEND)
        else:
            self._align_stable_start = 0.0

    def _handle_task_rec_descend(self, frame, dt: float, now: float) -> None:
        """TASK_REC_DESCEND：盲降到取货区（半速纠偏 + 注入下降速度）。"""
        if self._no_mcu_flight_test:
            if self._no_mcu_pickup_use_land_mode:
                self._land_and_transition(
                    FlightState.TRANS_DELIVERY,
                    "No-MCU pickup descent",
                )
                return

            if self._no_mcu_pickup_descend_start_time == 0.0:
                self._no_mcu_pickup_descend_start_time = now
                self._logger.info(
                    "No-MCU pickup guided descent start: "
                    f"vz={self._no_mcu_pickup_descend_vz_mps:.2f}m/s "
                    f"touchdown_alt={self._no_mcu_pickup_touchdown_alt_m:.2f}m"
                )

            if self._no_mcu_pickup_touchdown_reached(now):
                self._flight.send_body_velocity(0, 0, 0, 0)
                self._tick_vel = (0, 0, 0)
                self._controller.reset()
                self._logger.info("No-MCU pickup guided descent complete")
                self._transition_to(FlightState.TRANS_DELIVERY)
                return

            elapsed = now - self._no_mcu_pickup_descend_start_time
            if elapsed > self._no_mcu_pickup_descend_timeout_s:
                self._logger.error(
                    "No-MCU pickup guided descent timeout; landing and aborting mission"
                )
                self._land_and_transition(FlightState.IDLE, "No-MCU pickup descent timeout")
                return

            self._flight.send_body_velocity(
                0, 0, self._no_mcu_pickup_descend_vz_mps, 0
            )
            self._tick_vel = (0, 0, self._no_mcu_pickup_descend_vz_mps)
            return

        # 触地检测（优先检测）
        tel = self._flight.get_telemetry()
        if tel["alt"] < self._land_detect_alt or not tel["armed"]:
            self._controller.reset()
            self._transition_to(FlightState.TASK_REC_WAIT_LOAD)
            return

        vel = self._run_vision_pipeline(frame, dt, now)
        if vel is None:
            return

        vx, vy, vyaw = vel
        # 半速纠偏 + 注入下降速度
        cmd_vx = vx * self._DESCEND_GAIN
        cmd_vy = vy * self._DESCEND_GAIN
        self._flight.send_body_velocity(cmd_vx, cmd_vy, self._DESCEND_VZ, 0.0)
        self._tick_vel = (cmd_vx, cmd_vy, 0.0)

    def _handle_task_rec_wait_load(self, frame, dt: float, now: float) -> None:
        """TASK_REC_WAIT_LOAD：等待人工装填 + 红外确认。"""
        self._handle_mcu_action(
            command=MCUCommand.START_GRAB,
            success_response=MCUResponse.GRAB_DONE,
            fail_response=MCUResponse.GRAB_FAIL,
            timeout=self._grab_timeout,
            success_state=FlightState.TRANS_DELIVERY,
            now=now,
        )

    def _handle_no_mcu_guided_climb(self, now: float) -> bool:
        alt, tel = self._get_current_altitude()
        target_alt = self._transfer_alt
        threshold = self._takeoff_reached_threshold(target_alt)

        if not self._no_mcu_guided_climb_logged:
            self._no_mcu_guided_climb_logged = True
            self._mcu_cmd_time = now
            self._logger.info(
                "No-MCU guided climb start: "
                f"target_alt={target_alt:.2f}m threshold={threshold:.2f}m "
                f"vz=-{self._no_mcu_retakeoff_climb_vz_mps:.2f}m/s "
                f"mode={tel.get('mode')} armed={tel.get('armed')}"
            )

        if alt >= threshold:
            self._transfer_takeoff_done = True
            self._transfer_start_time = now
            self._flight.send_body_velocity(0, 0, 0, 0)
            self._tick_vel = (0, 0, 0)
            self._logger.info(
                "No-MCU guided climb complete: "
                f"alt={alt:.2f}m threshold={threshold:.2f}m"
            )
            return True

        if now - self._mcu_cmd_time > self._takeoff_timeout:
            self._logger.error(
                f"No-MCU guided climb timeout: {self._takeoff_timeout:.1f}s "
                f"target_alt={target_alt:.2f}m current_alt={alt:.2f}m "
                f"mode={tel.get('mode')} armed={tel.get('armed')}"
            )
            self._transition_to(FlightState.EMERGENCY)
            return True

        self._flight.send_body_velocity(
            0, 0, -self._no_mcu_retakeoff_climb_vz_mps, 0
        )
        self._tick_vel = (0, 0, -self._no_mcu_retakeoff_climb_vz_mps)
        return True

    def _handle_trans_delivery(self, frame, dt: float, now: float) -> None:
        """TRANS_DELIVERY：二次起飞 + 定距转移飞往投递区。"""
        if self._wait_no_mcu_post_land_ready(now):
            return

        # 子阶段 A：二次起飞
        if not self._transfer_takeoff_done:
            tel = self._flight.get_telemetry()
            if self._no_mcu_flight_test and bool(tel.get("armed", False)):
                self._handle_no_mcu_guided_climb(now)
                return

            if not self._mcu_cmd_sent:
                self._mcu_cmd_sent = True
                self._logger.info(
                    f"二次起飞命令：target_alt={self._transfer_alt:.2f}m"
                )
                success = self._flight.arm_and_takeoff(self._transfer_alt)
                now = time.time()
                self._mcu_cmd_time = now
                if success is False:
                    self._logger.error("二次起飞命令失败，进入 EMERGENCY")
                    self._transition_to(FlightState.EMERGENCY)
                    return

            if self._takeoff_altitude_reached(self._transfer_alt):
                self._transfer_takeoff_done = True
                self._transfer_start_time = now
                tel = self._flight.get_telemetry()
                self._logger.info(
                    f"二次起飞高度确认：alt={float(tel.get('alt', 0.0)):.2f}m "
                    f"threshold={self._takeoff_reached_threshold(self._transfer_alt):.2f}m"
                )
            elif now - self._mcu_cmd_time > self._takeoff_timeout:
                tel = self._flight.get_telemetry()
                self._logger.error(
                    f"二次起飞超时：{self._takeoff_timeout:.1f}s 未到达目标高度 "
                    f"{self._transfer_alt:.2f}m，当前 alt={float(tel.get('alt', 0.0)):.2f}m "
                    f"mode={tel.get('mode')} armed={tel.get('armed')}"
                )
                self._transition_to(FlightState.EMERGENCY)
            return

        # 子阶段 B：定距平飞
        self._flight.send_body_velocity(self._transfer_speed, 0, 0, 0)

        # 计算是否到达
        elapsed_flight = now - self._transfer_start_time
        expected_time = self._transfer_distance / self._transfer_speed
        if elapsed_flight >= expected_time:
            self._transition_to(FlightState.TASK_REL_ALIGN)

    def _handle_task_rel_align(self, frame, dt: float, now: float) -> None:
        """TASK_REL_ALIGN：视觉伺服对准投递区（cls_id=1）。"""
        vel = self._run_vision_pipeline(frame, dt, now)
        if vel is None:
            self._align_stable_start = 0.0
            return

        vel = self._apply_alignment_policy(vel)
        vx, vy, vyaw = vel
        self._tick_vel = vel

        self._flight.send_body_velocity(vx, vy, 0.0, vyaw)

        # 防抖跃迁判定
        if self._alignment_is_stable(vel):
            if self._align_stable_start == 0.0:
                self._align_stable_start = now
            elif now - self._align_stable_start >= self._align_hold_time:
                self._controller.reset()
                if self._no_mcu_flight_test:
                    self._land_and_transition(
                        FlightState.IDLE,
                        "No-MCU delivery alignment",
                    )
                else:
                    self._transition_to(FlightState.TASK_REL_DESCEND)
        else:
            self._align_stable_start = 0.0

    def _handle_task_rel_descend(self, frame, dt: float, now: float) -> None:
        """TASK_REL_DESCEND：盲降到投递区。"""
        if self._no_mcu_flight_test:
            self._land_and_transition(
                FlightState.IDLE,
                "No-MCU delivery descent",
            )
            return

        tel = self._flight.get_telemetry()
        if tel["alt"] < self._land_detect_alt or not tel["armed"]:
            self._controller.reset()
            self._transition_to(FlightState.TASK_REL_RELEASE)
            return

        vel = self._run_vision_pipeline(frame, dt, now)
        if vel is None:
            return

        vx, vy, vyaw = vel
        cmd_vx = vx * self._DESCEND_GAIN
        cmd_vy = vy * self._DESCEND_GAIN
        self._flight.send_body_velocity(cmd_vx, cmd_vy, self._DESCEND_VZ, 0.0)
        self._tick_vel = (cmd_vx, cmd_vy, 0.0)

    def _handle_task_rel_release(self, frame, dt: float, now: float) -> None:
        """TASK_REL_RELEASE：释放货物 + 红外校验。"""
        # Phase 2 单轮场景：释放成功后直接 OUTBOUND
        self._handle_mcu_action(
            command=MCUCommand.START_RELEASE,
            success_response=MCUResponse.RELEASE_DONE,
            fail_response=MCUResponse.RELEASE_FAIL,
            timeout=self._release_timeout,
            success_state=FlightState.OUTBOUND,
            now=now,
        )

    def _handle_trans_cargo(self, frame, dt: float, now: float) -> None:
        """TRANS_CARGO：飞往下一个取货区（Phase 2 简单实现）。"""
        # 子阶段 A：起飞
        if not self._transfer_takeoff_done:
            if not self._mcu_cmd_sent:
                self._mcu_cmd_sent = True
                success = self._flight.arm_and_takeoff(self._transfer_alt)
                now = time.time()
                self._mcu_cmd_time = now
                if success is False:
                    self._logger.error("TRANS_CARGO 起飞命令失败，进入 EMERGENCY")
                    self._transition_to(FlightState.EMERGENCY)
                    return

            if self._takeoff_altitude_reached(self._transfer_alt):
                self._transfer_takeoff_done = True
                self._transfer_start_time = now
            elif now - self._mcu_cmd_time > self._takeoff_timeout:
                self._logger.error(
                    f"TRANS_CARGO 起飞超时：{self._takeoff_timeout:.1f}s 未到达目标高度 {self._transfer_alt:.2f}m"
                )
                self._transition_to(FlightState.EMERGENCY)
            return

        # 子阶段 B：定距平飞（反向飞回取货区）
        self._flight.send_body_velocity(self._transfer_speed, 0, 0, 0)

        elapsed_flight = now - self._transfer_start_time
        expected_time = self._transfer_distance / self._transfer_speed
        if elapsed_flight >= expected_time:
            self._transition_to(FlightState.TASK_REC_ALIGN)

    def _handle_outbound(self, frame, dt: float, now: float) -> None:
        """OUTBOUND：返航降落。"""
        if not self._mcu_cmd_sent:
            self._mcu_cmd_sent = True
            # 直接执行降落
            success = self._flight.land()
            if success:
                self._transition_to(FlightState.IDLE)
            else:
                self._logger.error("返航降落失败，进入 EMERGENCY")
                self._transition_to(FlightState.EMERGENCY)

    def _handle_emergency(self, frame, dt: float, now: float) -> None:
        """EMERGENCY：紧急状态，交给飞控 Failsafe 接管。"""
        if not self._mcu_cmd_sent:
            self._mcu_cmd_sent = True
            self._logger.critical("进入紧急状态，切换 RTL")
            self._flight.set_mode(FlightMode.RTL)
        # 后续 tick 静默，由飞控 Failsafe 接管

    # ── 状态分派表（类级常量，定义在所有 handler 之后）──
    _STATE_HANDLERS = {
        FlightState.IDLE:               _handle_idle,
        FlightState.RESET:              _handle_reset,
        FlightState.INBOUND:            _handle_inbound,
        FlightState.TASK_REC_ALIGN:     _handle_task_rec_align,
        FlightState.TASK_REC_DESCEND:   _handle_task_rec_descend,
        FlightState.TASK_REC_WAIT_LOAD: _handle_task_rec_wait_load,
        FlightState.TRANS_DELIVERY:     _handle_trans_delivery,
        FlightState.TASK_REL_ALIGN:     _handle_task_rel_align,
        FlightState.TASK_REL_DESCEND:   _handle_task_rel_descend,
        FlightState.TASK_REL_RELEASE:   _handle_task_rel_release,
        FlightState.TRANS_CARGO:        _handle_trans_cargo,
        FlightState.OUTBOUND:           _handle_outbound,
        FlightState.EMERGENCY:          _handle_emergency,
    }
