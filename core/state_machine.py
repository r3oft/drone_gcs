"""
全局有限状态机（M5 模块）。
"""

import time
import logging
from enum import Enum

import numpy as np

from core.interfaces import (
    IFlightBridge, IMCUBridge,
    MCUCommand, MCUResponse, FlightMode,
)
from core.perception import TargetPoseEstimator
from core.servo_controller import VisualServoController
from utils.config_manager import ConfigManager
from utils.logger import setup_logger, FlightRecorder


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

        self._align_hold_time: float = config.get("fsm.align_hold_time_s", 1.5)
        self._target_lost_hover: float = config.get("fsm.target_lost_hover_s", 1.0)
        self._target_lost_climb: float = config.get("fsm.target_lost_climb_s", 3.0)
        self._climb_vz: float = config.get("fsm.climb_vz", -0.2)
        self._takeoff_alt: float = config.get("flight.takeoff_alt", 1.5)
        self._land_detect_alt: float = config.get("flight.land_detect_alt", 0.15)
        self._grab_timeout: float = config.get("mcu.grab_timeout_s", 10.0)
        self._release_timeout: float = config.get("mcu.release_timeout_s", 10.0)
        self._retry_max: int = config.get("mcu.retry_max", 2)
        self._transfer_speed: float = config.get("transfer.transfer_speed", 0.3)
        self._transfer_distance: float = config.get("transfer.delivery_distance_m", 3.0)
        self._transfer_alt: float = config.get("transfer.transfer_alt", 1.5)
        self._center_u: float = config.get("camera.center_u", 320)
        self._center_v: float = config.get("camera.center_v", 240)

        # ── FSM 核心状态 ──
        self._state: FlightState = FlightState.IDLE
        self._start_requested: bool = False
        self._stop_requested: bool = False

        # ── 防抖跃迁计时器 ──
        self._align_stable_start: float = 0.0

        # ── 目标丢失看门狗 ──
        self._last_target_seen: float = 0.0

        # ── MCU 交互状态 ──
        self._mcu_cmd_sent: bool = False
        self._mcu_cmd_time: float = 0.0
        self._mcu_retry_count: int = 0

        # ── 定距转移状态 ──
        self._transfer_start_time: float = 0.0
        self._transfer_takeoff_done: bool = False

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

        # ── 层 0：全局 Failsafe 拦截 ──
        if self._state != FlightState.EMERGENCY:
            self._check_failsafe()

        # 若 Failsafe 在上面触发了跃迁，此 tick 不再执行 handler
        if self._state == FlightState.EMERGENCY and \
           self._state != FlightState.EMERGENCY:
            pass  # 不会到达这里，逻辑保留供阅读

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
        self._state_enter_time = time.time()

        # 重置所有阶段性状态
        self._align_stable_start = 0.0
        self._mcu_cmd_sent = False
        self._mcu_retry_count = 0
        self._transfer_takeoff_done = False

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
            if self._state not in (FlightState.OUTBOUND, FlightState.EMERGENCY):
                self._logger.warning(
                    f"飞控模式被外部切换为 {tel['mode']}，进入 EMERGENCY"
                )
                self._transition_to(FlightState.EMERGENCY)

    def _run_vision_pipeline(
        self, frame: np.ndarray | None, dt: float, now: float
    ) -> tuple[float, float, float] | None:
        """
        执行视觉感知 + 看门狗检查 + 伺服控制律。

        Returns:
            (vx, vy, vyaw) 速度输出，或 None（目标丢失，已由内部处理悬停/爬升）
        """
        cls_id = self._get_target_cls_id()
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
        vx, vy, vyaw = self._controller.compute_velocity(
            target, self._center_u, self._center_v, dt
        )
        return (vx, vy, vyaw)

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
            self._mcu.send_command(command)
            self._mcu_cmd_sent = True
            self._mcu_cmd_time = now
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
                self._mcu.send_command(command)
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

    def _handle_idle(self, frame, dt: float, now: float) -> None:
        """IDLE：等待启动指令。"""
        if self._start_requested:
            self._start_requested = False
            self._transition_to(FlightState.RESET)

    def _handle_reset(self, frame, dt: float, now: float) -> None:
        """RESET：飞控连接 + MCU 复位 + 等待两者就绪。"""
        # 首次进入：执行连接和复位
        if not self._mcu_cmd_sent:
            self._flight.connect()
            self._mcu.send_command(MCUCommand.RESET)
            self._mcu_cmd_sent = True
            self._mcu_cmd_time = now
            return

        # 超时保护
        if now - self._state_enter_time > self._RESET_TIMEOUT_S:
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
        # arm_and_takeoff 是阻塞调用，首次进入即执行
        if not self._mcu_cmd_sent:
            self._mcu_cmd_sent = True  # 复用标志防止重复调用
            success = self._flight.arm_and_takeoff(self._takeoff_alt)
            if success:
                self._transition_to(FlightState.TASK_REC_ALIGN)
            else:
                self._logger.error("起飞失败，进入 EMERGENCY")
                self._transition_to(FlightState.EMERGENCY)

    def _handle_task_rec_align(self, frame, dt: float, now: float) -> None:
        """TASK_REC_ALIGN：视觉伺服对准取货区（cls_id=0）。"""
        vel = self._run_vision_pipeline(frame, dt, now)
        if vel is None:
            # 目标丢失，看门狗已在内部处理
            self._align_stable_start = 0.0
            return

        vx, vy, vyaw = vel
        self._tick_vel = vel

        # 下发速度指令（平面伺服，Vz=0）
        self._flight.send_body_velocity(vx, vy, 0.0, vyaw)

        # 防抖跃迁判定
        if vx == 0.0 and vy == 0.0 and vyaw == 0.0:
            if self._align_stable_start == 0.0:
                self._align_stable_start = now
            elif now - self._align_stable_start >= self._align_hold_time:
                self._controller.reset()
                self._transition_to(FlightState.TASK_REC_DESCEND)
        else:
            self._align_stable_start = 0.0

    def _handle_task_rec_descend(self, frame, dt: float, now: float) -> None:
        """TASK_REC_DESCEND：盲降到取货区（半速纠偏 + 注入下降速度）。"""
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

    def _handle_trans_delivery(self, frame, dt: float, now: float) -> None:
        """TRANS_DELIVERY：二次起飞 + 定距转移飞往投递区。"""
        # 子阶段 A：二次起飞
        if not self._transfer_takeoff_done:
            success = self._flight.arm_and_takeoff(self._transfer_alt)
            if success:
                self._transfer_takeoff_done = True
                self._transfer_start_time = now
            else:
                self._logger.error("二次起飞失败，进入 EMERGENCY")
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

        vx, vy, vyaw = vel
        self._tick_vel = vel

        self._flight.send_body_velocity(vx, vy, 0.0, vyaw)

        # 防抖跃迁判定
        if vx == 0.0 and vy == 0.0 and vyaw == 0.0:
            if self._align_stable_start == 0.0:
                self._align_stable_start = now
            elif now - self._align_stable_start >= self._align_hold_time:
                self._controller.reset()
                self._transition_to(FlightState.TASK_REL_DESCEND)
        else:
            self._align_stable_start = 0.0

    def _handle_task_rel_descend(self, frame, dt: float, now: float) -> None:
        """TASK_REL_DESCEND：盲降到投递区。"""
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
            success = self._flight.arm_and_takeoff(self._transfer_alt)
            if success:
                self._transfer_takeoff_done = True
                self._transfer_start_time = now
            else:
                self._logger.error("TRANS_CARGO 起飞失败，进入 EMERGENCY")
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
