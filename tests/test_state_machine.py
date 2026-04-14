"""
core/state_machine.py 单元测试。

所有测试使用内联 Stub 替身，不依赖 GPU / YOLO / dronekit。
"""

import time
import pytest

from core.state_machine import FlightState, GlobalFSM
from core.interfaces import MCUCommand, MCUResponse, FlightMode


# ═══════════════════════════════════════════════════════
#  Stub 替身（轻量级测试 Mock）
# ═══════════════════════════════════════════════════════

class StubFlightBridge:
    """飞控接口 Stub，记录所有调用并提供可编程返回值。"""

    def __init__(self):
        self._connected = True
        self._telemetry = {
            "armed": True,
            "mode": "GUIDED",
            "alt": 1.5,
            "heading": 0.0,
            "battery_pct": 0.9,
            "heartbeat_ok": True,
        }
        self.velocity_log: list[tuple] = []
        self.connect_return = True
        self.takeoff_return = True
        self.land_return = True
        self.set_mode_return = True
        self.set_mode_calls: list[str] = []

    def connect(self) -> bool:
        return self.connect_return

    def arm_and_takeoff(self, target_alt: float) -> bool:
        return self.takeoff_return

    def send_body_velocity(self, vx, vy, vz, yaw_rate) -> None:
        self.velocity_log.append((vx, vy, vz, yaw_rate))

    def land(self) -> bool:
        return self.land_return

    def set_mode(self, mode: str) -> bool:
        self.set_mode_calls.append(mode)
        return self.set_mode_return

    def get_telemetry(self) -> dict:
        return dict(self._telemetry)

    def is_connected(self) -> bool:
        return self._connected


class StubMCUBridge:
    """MCU 接口 Stub，支持预编程响应队列。"""

    def __init__(self):
        self._responses: list[str | None] = []
        self._response_idx = 0
        self.command_log: list[str] = []
        self._connected = True

    def set_responses(self, *responses):
        """预设响应序列，每次 get_latest_response() 消费一个。"""
        self._responses = list(responses)
        self._response_idx = 0

    def send_command(self, command: str) -> bool:
        self.command_log.append(command)
        return True

    def get_latest_response(self) -> str | None:
        if self._response_idx < len(self._responses):
            resp = self._responses[self._response_idx]
            self._response_idx += 1
            return resp
        return None

    def is_connected(self) -> bool:
        return self._connected


class StubPerception:
    """感知模块 Stub，返回预设的目标或 None。"""

    def __init__(self):
        self.target: dict | None = {
            "u": 320.0, "v": 240.0,
            "theta": 0.0, "conf": 0.9,
            "w": 30.0, "h": 30.0,
        }

    def process_frame(self, frame, target_cls_id):
        return self.target


class StubController:
    """伺服控制器 Stub，返回预编程的速度输出。"""

    def __init__(self):
        self.velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.reset_count = 0

    def compute_velocity(self, target_pose, center_u, center_v, dt):
        return self.velocity

    def reset(self):
        self.reset_count += 1


class StubConfig:
    """配置管理器 Stub，返回默认值。"""

    _DEFAULTS = {
        "fsm.align_hold_time_s": 1.5,
        "fsm.target_lost_hover_s": 1.0,
        "fsm.target_lost_climb_s": 3.0,
        "fsm.climb_vz": -0.2,
        "flight.takeoff_alt": 1.5,
        "flight.land_detect_alt": 0.15,
        "mcu.grab_timeout_s": 10.0,
        "mcu.release_timeout_s": 10.0,
        "mcu.retry_max": 2,
        "transfer.transfer_speed": 0.3,
        "transfer.delivery_distance_m": 3.0,
        "transfer.transfer_alt": 1.5,
        "camera.center_u": 320,
        "camera.center_v": 240,
        "logging.level": "WARNING",  # 抑制测试中的日志输出
        "logging.log_dir": "logs/",
        "logging.enable_flight_recorder": False,  # 测试中禁用黑匣子
    }

    def get(self, key, default=None):
        return self._DEFAULTS.get(key, default)


# ═══════════════════════════════════════════════════════
#  Fixture
# ═══════════════════════════════════════════════════════

@pytest.fixture
def flight():
    return StubFlightBridge()

@pytest.fixture
def mcu():
    return StubMCUBridge()

@pytest.fixture
def perception():
    return StubPerception()

@pytest.fixture
def controller():
    return StubController()

@pytest.fixture
def config():
    return StubConfig()

@pytest.fixture
def fsm(flight, mcu, perception, controller, config):
    return GlobalFSM(flight, mcu, perception, controller, config)


# ═══════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════

def advance_to_state(fsm, flight, mcu, target_state: FlightState):
    """
    快速推进 FSM 到指定状态（用于跳过前置状态的测试设置）。
    直接设置内部状态，绕过正常跃迁流程。
    """
    fsm._state = target_state
    fsm._state_enter_time = time.time()
    fsm._mcu_cmd_sent = False
    fsm._mcu_retry_count = 0
    fsm._align_stable_start = 0.0
    fsm._transfer_takeoff_done = False
    fsm._last_target_seen = time.time()


# ═══════════════════════════════════════════════════════
#  测试用例
# ═══════════════════════════════════════════════════════

class TestInitialState:
    """测试 #1: 构造后处于 IDLE 状态。"""

    def test_initial_state_is_idle(self, fsm):
        assert fsm.state == FlightState.IDLE


class TestRequestStart:
    """测试 #2: request_start → IDLE 到 RESET。"""

    def test_request_start_transitions_to_reset(self, fsm):
        fsm.request_start()
        assert fsm.state == FlightState.IDLE  # 标志设置，还未跃迁
        fsm.tick(None)  # 执行 _handle_idle
        assert fsm.state == FlightState.RESET

    def test_request_start_ignored_in_non_idle(self, fsm):
        fsm._state = FlightState.INBOUND
        fsm.request_start()
        assert fsm.state == FlightState.INBOUND


class TestResetToInbound:
    """测试 #3: RESET → INBOUND。"""

    def test_reset_to_inbound(self, fsm, flight, mcu):
        # 进入 RESET
        fsm.request_start()
        fsm.tick(None)
        assert fsm.state == FlightState.RESET

        # 第一个 tick: 执行连接 + MCU RESET
        fsm.tick(None)
        assert MCUCommand.RESET in mcu.command_log

        # 第二个 tick: 心跳 OK + RESET_DONE → INBOUND
        mcu.set_responses(MCUResponse.RESET_DONE)
        fsm.tick(None)
        assert fsm.state == FlightState.INBOUND

    def test_reset_timeout_to_emergency(self, fsm, flight, mcu):
        fsm.request_start()
        fsm.tick(None)
        assert fsm.state == FlightState.RESET

        # 首次 tick 触发连接
        fsm.tick(None)

        # 模拟超时：将进入时间设到很久以前
        fsm._state_enter_time = time.time() - 60
        fsm.tick(None)
        assert fsm.state == FlightState.EMERGENCY


class TestInboundToAlign:
    """测试 #4: INBOUND → TASK_REC_ALIGN。"""

    def test_inbound_to_task_rec_align(self, fsm, flight):
        advance_to_state(fsm, flight, None, FlightState.INBOUND)
        flight.takeoff_return = True
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REC_ALIGN

    def test_inbound_takeoff_fail(self, fsm, flight):
        advance_to_state(fsm, flight, None, FlightState.INBOUND)
        flight.takeoff_return = False
        fsm.tick(None)
        assert fsm.state == FlightState.EMERGENCY


class TestAlignDebounce:
    """测试 #5 & #6: 防抖跃迁。"""

    def test_align_debounce_transition(self, fsm, flight, controller):
        """速度连续 1.5s 为零 → TASK_REC_DESCEND。"""
        advance_to_state(fsm, flight, None, FlightState.TASK_REC_ALIGN)
        controller.velocity = (0.0, 0.0, 0.0)

        # 第一个 tick：启动计时器
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REC_ALIGN
        assert fsm._align_stable_start > 0

        # 模拟 1.5s 前已经开始稳定
        fsm._align_stable_start = time.time() - 2.0
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REC_DESCEND
        assert controller.reset_count >= 1

    def test_align_debounce_reset(self, fsm, flight, controller):
        """速度中途非零 → 计时器重置。"""
        advance_to_state(fsm, flight, None, FlightState.TASK_REC_ALIGN)
        controller.velocity = (0.0, 0.0, 0.0)

        # 启动计时器
        fsm.tick(None)
        assert fsm._align_stable_start > 0

        # 速度变非零
        controller.velocity = (0.1, 0.0, 0.0)
        fsm.tick(None)
        assert fsm._align_stable_start == 0.0
        assert fsm.state == FlightState.TASK_REC_ALIGN


class TestTargetLost:
    """测试 #7 & #8: 目标丢失看门狗。"""

    def test_target_lost_hover(self, fsm, flight, perception):
        """丢失 1~3s → 悬停（全零速度）。"""
        advance_to_state(fsm, flight, None, FlightState.TASK_REC_ALIGN)
        perception.target = None

        # 模拟 1.5s 前最后看到目标
        fsm._last_target_seen = time.time() - 1.5
        flight.velocity_log.clear()
        fsm.tick(None)

        assert len(flight.velocity_log) == 1
        assert flight.velocity_log[0] == (0, 0, 0, 0)

    def test_target_lost_climb(self, fsm, flight, perception):
        """丢失 >3s → 爬升搜索。"""
        advance_to_state(fsm, flight, None, FlightState.TASK_REC_ALIGN)
        perception.target = None

        # 模拟 4s 前最后看到目标
        fsm._last_target_seen = time.time() - 4.0
        flight.velocity_log.clear()
        fsm.tick(None)

        assert len(flight.velocity_log) == 1
        vx, vy, vz, yaw = flight.velocity_log[0]
        assert vx == 0 and vy == 0
        assert vz == -0.2  # climb_vz
        assert yaw == 0


class TestDescendToWaitLoad:
    """测试 #9: 高度低于阈值 → TASK_REC_WAIT_LOAD。"""

    def test_descend_to_wait_load(self, fsm, flight):
        advance_to_state(fsm, flight, None, FlightState.TASK_REC_DESCEND)
        flight._telemetry["alt"] = 0.1  # 低于 land_detect_alt (0.15)
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REC_WAIT_LOAD

    def test_descend_continues_above_threshold(self, fsm, flight, controller):
        advance_to_state(fsm, flight, None, FlightState.TASK_REC_DESCEND)
        flight._telemetry["alt"] = 1.0
        controller.velocity = (0.1, 0.05, 0.0)
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REC_DESCEND
        # 验证半速纠偏 + 下降速度
        assert len(flight.velocity_log) > 0
        vx, vy, vz, yaw = flight.velocity_log[-1]
        assert vx == pytest.approx(0.05)   # 0.1 * 0.5
        assert vy == pytest.approx(0.025)  # 0.05 * 0.5
        assert vz == pytest.approx(0.15)   # DESCEND_VZ


class TestGrabFlow:
    """测试 #10~#13: MCU 抓取交互。"""

    def test_grab_success(self, fsm, flight, mcu):
        """MCU 返回 GRAB_DONE → TRANS_DELIVERY。"""
        advance_to_state(fsm, flight, mcu, FlightState.TASK_REC_WAIT_LOAD)

        # 首次 tick: 发送 START_GRAB
        fsm.tick(None)
        assert MCUCommand.START_GRAB in mcu.command_log

        # 第二次 tick: 收到 GRAB_DONE
        mcu.set_responses(MCUResponse.GRAB_DONE)
        fsm.tick(None)
        assert fsm.state == FlightState.TRANS_DELIVERY

    def test_grab_retry_then_success(self, fsm, flight, mcu):
        """GRAB_FAIL → 重试 → GRAB_DONE。"""
        advance_to_state(fsm, flight, mcu, FlightState.TASK_REC_WAIT_LOAD)

        # 首次 tick: 发送指令
        fsm.tick(None)

        # 第二次 tick: 收到 GRAB_FAIL → 重试
        mcu.set_responses(MCUResponse.GRAB_FAIL)
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REC_WAIT_LOAD
        assert fsm._mcu_retry_count == 1

        # 第三次 tick: 收到 GRAB_DONE
        mcu.set_responses(MCUResponse.GRAB_DONE)
        fsm.tick(None)
        assert fsm.state == FlightState.TRANS_DELIVERY

    def test_grab_retry_exhausted(self, fsm, flight, mcu):
        """超过重试次数 → EMERGENCY。"""
        advance_to_state(fsm, flight, mcu, FlightState.TASK_REC_WAIT_LOAD)

        # 首次 tick: 发送指令
        fsm.tick(None)

        # 连续失败直到重试耗尽（retry_max=2）
        for i in range(3):
            mcu.set_responses(MCUResponse.GRAB_FAIL)
            fsm.tick(None)
            if fsm.state == FlightState.EMERGENCY:
                break

        assert fsm.state == FlightState.EMERGENCY

    def test_grab_timeout_retry(self, fsm, flight, mcu):
        """超时无响应 → 重试。"""
        advance_to_state(fsm, flight, mcu, FlightState.TASK_REC_WAIT_LOAD)

        # 首次 tick: 发送指令
        fsm.tick(None)

        # 模拟超时
        fsm._mcu_cmd_time = time.time() - 15  # 超过 grab_timeout_s (10)
        mcu.set_responses(None)
        fsm.tick(None)
        assert fsm._mcu_retry_count == 1


class TestTransDelivery:
    """测试 #14: 定距转移完成 → TASK_REL_ALIGN。"""

    def test_trans_delivery_to_rel_align(self, fsm, flight, mcu):
        advance_to_state(fsm, flight, mcu, FlightState.TRANS_DELIVERY)

        # 首次 tick: 二次起飞
        fsm.tick(None)
        assert fsm._transfer_takeoff_done is True

        # 模拟飞行时间足够（distance=3.0, speed=0.3, 需要 10s）
        fsm._transfer_start_time = time.time() - 11
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REL_ALIGN


class TestReleaseFlow:
    """测试 #15: RELEASE_DONE → OUTBOUND。"""

    def test_release_success(self, fsm, flight, mcu):
        advance_to_state(fsm, flight, mcu, FlightState.TASK_REL_RELEASE)

        # 首次 tick: 发送 START_RELEASE
        fsm.tick(None)
        assert MCUCommand.START_RELEASE in mcu.command_log

        # 第二次 tick: 收到 RELEASE_DONE
        mcu.set_responses(MCUResponse.RELEASE_DONE)
        fsm.tick(None)
        assert fsm.state == FlightState.OUTBOUND


class TestOutbound:
    """测试 #16: land() 成功 → IDLE。"""

    def test_outbound_to_idle(self, fsm, flight, mcu):
        advance_to_state(fsm, flight, mcu, FlightState.OUTBOUND)
        flight.land_return = True
        fsm.tick(None)
        assert fsm.state == FlightState.IDLE

    def test_outbound_land_fail(self, fsm, flight, mcu):
        advance_to_state(fsm, flight, mcu, FlightState.OUTBOUND)
        flight.land_return = False
        fsm.tick(None)
        assert fsm.state == FlightState.EMERGENCY


class TestFailsafe:
    """测试 #17 & #18: Failsafe 检测。"""

    def test_failsafe_heartbeat_lost(self, fsm, flight):
        advance_to_state(fsm, flight, None, FlightState.TASK_REC_ALIGN)
        flight._telemetry["heartbeat_ok"] = False
        fsm.tick(None)
        assert fsm.state == FlightState.EMERGENCY

    def test_failsafe_connection_lost(self, fsm, flight):
        advance_to_state(fsm, flight, None, FlightState.TASK_REC_ALIGN)
        flight._connected = False
        fsm.tick(None)
        assert fsm.state == FlightState.EMERGENCY

    def test_failsafe_mode_override(self, fsm, flight):
        advance_to_state(fsm, flight, None, FlightState.TASK_REC_ALIGN)
        flight._telemetry["mode"] = FlightMode.RTL
        fsm.tick(None)
        assert fsm.state == FlightState.EMERGENCY

    def test_failsafe_not_triggered_in_outbound(self, fsm, flight, mcu):
        """OUTBOUND 状态下 RTL 模式不触发 EMERGENCY。"""
        advance_to_state(fsm, flight, mcu, FlightState.OUTBOUND)
        flight._telemetry["mode"] = FlightMode.RTL
        flight.land_return = True
        fsm.tick(None)
        # 不应该进入 EMERGENCY，而是正常处理 OUTBOUND
        assert fsm.state != FlightState.EMERGENCY or fsm.state == FlightState.IDLE


class TestRequestStop:
    """测试 #19: request_stop → OUTBOUND。"""

    def test_request_stop(self, fsm, flight):
        advance_to_state(fsm, flight, None, FlightState.TASK_REC_ALIGN)
        fsm.request_stop()
        assert fsm.state == FlightState.OUTBOUND

    def test_request_stop_ignored_in_idle(self, fsm):
        fsm.request_stop()
        assert fsm.state == FlightState.IDLE

    def test_request_stop_ignored_in_emergency(self, fsm):
        fsm._state = FlightState.EMERGENCY
        fsm.request_stop()
        assert fsm.state == FlightState.EMERGENCY


class TestEmergencyHandler:
    """EMERGENCY 状态 handler 行为。"""

    def test_emergency_sets_rtl(self, fsm, flight, mcu):
        advance_to_state(fsm, flight, mcu, FlightState.EMERGENCY)
        # 需要跳过 failsafe 检查（因为 EMERGENCY 状态不检查）
        fsm.tick(None)
        assert FlightMode.RTL in flight.set_mode_calls

    def test_emergency_only_calls_rtl_once(self, fsm, flight, mcu):
        advance_to_state(fsm, flight, mcu, FlightState.EMERGENCY)
        fsm.tick(None)
        fsm.tick(None)
        fsm.tick(None)
        assert flight.set_mode_calls.count(FlightMode.RTL) == 1


class TestFullHappyPath:
    """测试 #20: 完整单轮流程 E2E。"""

    def test_full_happy_path(self, fsm, flight, mcu, perception, controller):
        # IDLE → RESET
        fsm.request_start()
        fsm.tick(None)
        assert fsm.state == FlightState.RESET

        # RESET: 连接 + MCU 复位
        fsm.tick(None)
        mcu.set_responses(MCUResponse.RESET_DONE)
        fsm.tick(None)
        assert fsm.state == FlightState.INBOUND

        # INBOUND → TASK_REC_ALIGN
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REC_ALIGN

        # TASK_REC_ALIGN: 速度为零，触发防抖跃迁
        controller.velocity = (0.0, 0.0, 0.0)
        fsm.tick(None)  # 启动计时
        fsm._align_stable_start = time.time() - 2.0  # 模拟已稳定 2s
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REC_DESCEND

        # TASK_REC_DESCEND → TASK_REC_WAIT_LOAD
        flight._telemetry["alt"] = 0.1
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REC_WAIT_LOAD

        # TASK_REC_WAIT_LOAD: START_GRAB → GRAB_DONE
        fsm.tick(None)
        mcu.set_responses(MCUResponse.GRAB_DONE)
        fsm.tick(None)
        assert fsm.state == FlightState.TRANS_DELIVERY

        # TRANS_DELIVERY: 起飞 + 定距转移
        flight._telemetry["alt"] = 1.5
        fsm.tick(None)
        assert fsm._transfer_takeoff_done is True
        fsm._transfer_start_time = time.time() - 11  # 模拟飞行 11s
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REL_ALIGN

        # TASK_REL_ALIGN → TASK_REL_DESCEND
        controller.velocity = (0.0, 0.0, 0.0)
        fsm._last_target_seen = time.time()
        fsm.tick(None)
        fsm._align_stable_start = time.time() - 2.0
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REL_DESCEND

        # TASK_REL_DESCEND → TASK_REL_RELEASE
        flight._telemetry["alt"] = 0.1
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REL_RELEASE

        # TASK_REL_RELEASE: START_RELEASE → RELEASE_DONE
        fsm.tick(None)
        mcu.set_responses(MCUResponse.RELEASE_DONE)
        fsm.tick(None)
        assert fsm.state == FlightState.OUTBOUND

        # OUTBOUND → IDLE
        flight.land_return = True
        fsm.tick(None)
        assert fsm.state == FlightState.IDLE


class TestRelAlignDescendSymmetry:
    """验证投递阶段与取货阶段的对称性。"""

    def test_rel_align_sends_correct_cls_id(self, fsm, flight, perception):
        """TASK_REL_ALIGN 状态下应使用 cls_id=1。"""
        advance_to_state(fsm, flight, None, FlightState.TASK_REL_ALIGN)

        # 跟踪 perception.process_frame 的 cls_id 参数
        calls = []
        original_pf = perception.process_frame

        def tracking_pf(frame, cls_id):
            calls.append(cls_id)
            return original_pf(frame, cls_id)

        perception.process_frame = tracking_pf
        fsm.tick(None)
        assert 1 in calls

    def test_rel_descend_to_release(self, fsm, flight):
        """TASK_REL_DESCEND 触地 → TASK_REL_RELEASE。"""
        advance_to_state(fsm, flight, None, FlightState.TASK_REL_DESCEND)
        flight._telemetry["alt"] = 0.1
        fsm.tick(None)
        assert fsm.state == FlightState.TASK_REL_RELEASE
