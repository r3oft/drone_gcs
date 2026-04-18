"""
utils/mock.py 单元测试。

测试 MockStreamer、MockFlightBridge、MockMCUBridge 三个模拟替身的
接口一致性、模拟行为正确性和测试辅助功能。
"""

import os
import time
import tempfile
import threading

import cv2
import numpy as np
import pytest

from utils.mock import MockStreamer, MockFlightBridge, MockMCUBridge
from core.interfaces import (
    IFlightBridge, IMCUBridge, IStreamer,
    FlightMode, MCUCommand, MCUResponse,
)


# ═══════════════════════════════════════════════════════
#  Fixtures — 测试数据准备
# ═══════════════════════════════════════════════════════

@pytest.fixture
def tmp_image_dir(tmp_path):
    """创建临时图片目录，含 3 张测试图片。"""
    for i in range(3):
        img = np.full((48, 64, 3), fill_value=(i * 80), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{i:04d}.png"), img)
    return str(tmp_path)


@pytest.fixture
def tmp_video_file(tmp_path):
    """创建临时视频文件，含 5 帧。"""
    video_path = str(tmp_path / "test.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(video_path, fourcc, 10, (64, 48))
    for i in range(5):
        frame = np.full((48, 64, 3), fill_value=(i * 50), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


@pytest.fixture
def flight():
    """创建 MockFlightBridge 实例。"""
    return MockFlightBridge()


@pytest.fixture
def mcu():
    """创建 MockMCUBridge 实例。"""
    return MockMCUBridge()


# ═══════════════════════════════════════════════════════
#  TestMockStreamer — 视频流模拟
# ═══════════════════════════════════════════════════════

class TestMockStreamerVideo:
    """视频文件模式测试。"""

    def test_video_file_returns_frame(self, tmp_video_file):
        """视频文件模式：get_latest_frame 返回 ndarray。"""
        streamer = MockStreamer(tmp_video_file)
        assert streamer.is_opened()
        frame = streamer.get_latest_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (48, 64, 3)
        streamer.release()

    def test_video_file_loop(self, tmp_video_file):
        """视频循环播放：读完后重新从头开始。"""
        streamer = MockStreamer(tmp_video_file, loop=True)
        # 读完 5 帧 + 1 帧触发循环
        for _ in range(6):
            frame = streamer.get_latest_frame()
            assert frame is not None
        streamer.release()

    def test_video_file_no_loop(self, tmp_video_file):
        """视频非循环：读完后返回 None。"""
        streamer = MockStreamer(tmp_video_file, loop=False)
        frames_read = 0
        for _ in range(10):
            frame = streamer.get_latest_frame()
            if frame is None:
                break
            frames_read += 1
        assert frames_read == 5
        assert not streamer.is_opened()
        streamer.release()


class TestMockStreamerImage:
    """图片目录模式测试。"""

    def test_image_dir_returns_frames(self, tmp_image_dir):
        """图片目录模式：按顺序返回帧。"""
        streamer = MockStreamer(tmp_image_dir)
        assert streamer.is_opened()
        frames = []
        for _ in range(3):
            frame = streamer.get_latest_frame()
            assert frame is not None
            frames.append(frame)
        # 各帧应有不同的像素值
        assert not np.array_equal(frames[0], frames[1])
        streamer.release()

    def test_image_dir_loop(self, tmp_image_dir):
        """图片循环播放：到达末尾后重新开始。"""
        streamer = MockStreamer(tmp_image_dir, loop=True)
        # 读 3 帧 + 再读 1 帧应循环
        for _ in range(4):
            frame = streamer.get_latest_frame()
            assert frame is not None
        streamer.release()

    def test_image_dir_no_loop(self, tmp_image_dir):
        """图片非循环：到达末尾后返回 None。"""
        streamer = MockStreamer(tmp_image_dir, loop=False)
        frames_read = 0
        for _ in range(5):
            frame = streamer.get_latest_frame()
            if frame is None:
                break
            frames_read += 1
        assert frames_read == 3
        assert not streamer.is_opened()
        streamer.release()


class TestMockStreamerGeneral:
    """通用功能测试。"""

    def test_invalid_source_raises(self):
        """无效路径抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            MockStreamer("/nonexistent/path/to/nowhere")

    def test_release_closes_stream(self, tmp_image_dir):
        """release 后 is_opened 返回 False。"""
        streamer = MockStreamer(tmp_image_dir)
        assert streamer.is_opened()
        streamer.release()
        assert not streamer.is_opened()

    def test_release_then_get_frame_returns_none(self, tmp_image_dir):
        """release 后 get_latest_frame 返回 None。"""
        streamer = MockStreamer(tmp_image_dir)
        streamer.release()
        assert streamer.get_latest_frame() is None

    def test_isinstance_istreamer(self, tmp_image_dir):
        """MockStreamer 是 IStreamer 的实例。"""
        streamer = MockStreamer(tmp_image_dir)
        assert isinstance(streamer, IStreamer)
        streamer.release()


# ═══════════════════════════════════════════════════════
#  TestMockFlightBridge — 飞控模拟
# ═══════════════════════════════════════════════════════

class TestMockFlightBridgeConnect:
    """连接相关测试。"""

    def test_initial_not_connected(self, flight):
        """初始状态未连接。"""
        assert not flight.is_connected()

    def test_connect_returns_true(self, flight):
        """connect 返回 True。"""
        assert flight.connect()

    def test_connect_sets_connected(self, flight):
        """connect 后 is_connected 返回 True。"""
        flight.connect()
        assert flight.is_connected()
        # heartbeat_ok 也应设为 True
        telem = flight.get_telemetry()
        assert telem["heartbeat_ok"]

    def test_isinstance_iflightbridge(self, flight):
        """MockFlightBridge 是 IFlightBridge 的实例。"""
        assert isinstance(flight, IFlightBridge)


class TestMockFlightBridgeFlight:
    """飞行行为测试。"""

    def test_arm_and_takeoff(self, flight):
        """arm_and_takeoff 后 alt 和 armed 状态正确。"""
        flight.connect()
        result = flight.arm_and_takeoff(1.5)
        assert result is True
        telem = flight.get_telemetry()
        assert telem["armed"] is True
        assert telem["alt"] == pytest.approx(1.5)
        assert telem["mode"] == FlightMode.GUIDED

    def test_velocity_logging(self, flight):
        """send_body_velocity 记录到 velocity_log。"""
        flight.connect()
        flight.arm_and_takeoff(2.0)
        flight.send_body_velocity(0.1, -0.05, 0.0, 0.3)
        log = flight.get_velocity_log()
        assert len(log) == 1
        assert log[0]["vx"] == pytest.approx(0.1)
        assert log[0]["vy"] == pytest.approx(-0.05)
        assert log[0]["vz"] == pytest.approx(0.0)
        assert log[0]["yaw_rate"] == pytest.approx(0.3)
        assert "time" in log[0]

    def test_alt_simulation_descend(self, flight):
        """下发 vz>0 后高度递减。"""
        flight.connect()
        flight.arm_and_takeoff(1.5)
        # 多次下发 vz=0.15 (向下)，高度应逐步递减
        for _ in range(10):
            flight.send_body_velocity(0.0, 0.0, 0.15, 0.0)
        telem = flight.get_telemetry()
        assert telem["alt"] < 1.5
        assert telem["alt"] >= 0.0

    def test_alt_simulation_climb(self, flight):
        """下发 vz<0 后高度递增。"""
        flight.connect()
        flight.arm_and_takeoff(1.0)
        flight.send_body_velocity(0.0, 0.0, -0.2, 0.0)
        telem = flight.get_telemetry()
        assert telem["alt"] > 1.0

    def test_alt_floor_at_zero(self, flight):
        """高度不会低于 0。"""
        flight.connect()
        flight.arm_and_takeoff(0.1)
        for _ in range(100):
            flight.send_body_velocity(0.0, 0.0, 1.0, 0.0)
        telem = flight.get_telemetry()
        assert telem["alt"] == pytest.approx(0.0)

    def test_land(self, flight):
        """land 后 alt=0, armed=False。"""
        flight.connect()
        flight.arm_and_takeoff(1.5)
        result = flight.land()
        assert result is True
        telem = flight.get_telemetry()
        assert telem["alt"] == pytest.approx(0.0)
        assert telem["armed"] is False
        assert telem["mode"] == FlightMode.LAND

    def test_set_mode(self, flight):
        """set_mode 更新 telemetry mode。"""
        flight.connect()
        flight.set_mode(FlightMode.LOITER)
        assert flight.get_telemetry()["mode"] == FlightMode.LOITER
        flight.set_mode(FlightMode.RTL)
        assert flight.get_telemetry()["mode"] == FlightMode.RTL

    def test_heartbeat_control(self, flight):
        """set_heartbeat_ok(False) → telemetry 反映。"""
        flight.connect()
        assert flight.get_telemetry()["heartbeat_ok"] is True
        flight.set_heartbeat_ok(False)
        assert flight.get_telemetry()["heartbeat_ok"] is False

    def test_set_connected(self, flight):
        """手动控制连接状态。"""
        flight.connect()
        assert flight.is_connected()
        flight.set_connected(False)
        assert not flight.is_connected()

    def test_command_log(self, flight):
        """所有方法调用被记录到 command_log。"""
        flight.connect()
        flight.arm_and_takeoff(1.5)
        flight.set_mode(FlightMode.LOITER)
        flight.land()
        log = flight.get_command_log()
        actions = [entry["action"] for entry in log]
        assert actions == ["connect", "arm_and_takeoff", "set_mode", "land"]


# ═══════════════════════════════════════════════════════
#  TestMockMCUBridge — MCU 通信模拟
# ═══════════════════════════════════════════════════════

class TestMockMCUBridgeDefault:
    """默认预设测试。"""

    def test_default_responses_preset(self, mcu):
        """构造后默认预设 RESET_DONE / GRAB_DONE / RELEASE_DONE。"""
        # 发 RESET → 应有 RESET_DONE 响应（延迟 0.5s）
        mcu.set_auto_response(MCUCommand.RESET, MCUResponse.RESET_DONE, delay_s=0.0)
        mcu.send_command(MCUCommand.RESET)
        resp = mcu.get_latest_response()
        assert resp == MCUResponse.RESET_DONE

    def test_isinstance_imcubridge(self, mcu):
        """MockMCUBridge 是 IMCUBridge 的实例。"""
        assert isinstance(mcu, IMCUBridge)

    def test_initial_connected(self, mcu):
        """初始状态已连接。"""
        assert mcu.is_connected()


class TestMockMCUBridgeAutoResponse:
    """自动响应测试。"""

    def test_auto_response_immediate(self, mcu):
        """delay=0 时 send_command 后立即可读取响应。"""
        mcu.set_auto_response("TEST_CMD", "TEST_RESP", delay_s=0.0)
        mcu.send_command("TEST_CMD")
        resp = mcu.get_latest_response()
        assert resp == "TEST_RESP"

    def test_auto_response_delay(self, mcu):
        """延迟响应：send_command 后立即无响应，等待后有响应。"""
        mcu.set_auto_response("SLOW_CMD", "SLOW_RESP", delay_s=0.3)
        mcu.send_command("SLOW_CMD")
        # 立即读取应为 None
        resp_immediate = mcu.get_latest_response()
        assert resp_immediate is None
        # 等待足够时间后读取
        time.sleep(0.5)
        resp_after = mcu.get_latest_response()
        assert resp_after == "SLOW_RESP"

    def test_response_consumed(self, mcu):
        """get_latest_response 返回后清空，再次返回 None。"""
        mcu.set_auto_response("CMD", "RESP", delay_s=0.0)
        mcu.send_command("CMD")
        assert mcu.get_latest_response() == "RESP"
        assert mcu.get_latest_response() is None

    def test_clear_auto_responses(self, mcu):
        """clear 后 send_command 不再产生自动响应。"""
        mcu.clear_auto_responses()
        mcu.send_command(MCUCommand.START_GRAB)
        time.sleep(0.1)
        assert mcu.get_latest_response() is None


class TestMockMCUBridgeFailure:
    """故障注入测试。"""

    def test_inject_failure_once(self, mcu):
        """注入故障后首次返回故障响应，第二次恢复正常。"""
        mcu.set_auto_response(MCUCommand.START_GRAB, MCUResponse.GRAB_DONE, delay_s=0.0)
        mcu.inject_failure(MCUCommand.START_GRAB, MCUResponse.GRAB_FAIL)

        # 第一次：故障注入
        mcu.send_command(MCUCommand.START_GRAB)
        resp1 = mcu.get_latest_response()
        assert resp1 == MCUResponse.GRAB_FAIL

        # 第二次：恢复正常
        mcu.send_command(MCUCommand.START_GRAB)
        resp2 = mcu.get_latest_response()
        assert resp2 == MCUResponse.GRAB_DONE


class TestMockMCUBridgeLog:
    """指令日志测试。"""

    def test_command_log(self, mcu):
        """所有指令被正确记录。"""
        mcu.send_command(MCUCommand.RESET)
        mcu.send_command(MCUCommand.START_GRAB)
        mcu.send_command(MCUCommand.START_RELEASE)
        log = mcu.get_command_log()
        assert len(log) == 3
        cmds = [entry["command"] for entry in log]
        assert cmds == [MCUCommand.RESET, MCUCommand.START_GRAB, MCUCommand.START_RELEASE]
        # 每条记录都有时间戳
        for entry in log:
            assert "time" in entry

    def test_set_connected(self, mcu):
        """手动控制连接状态。"""
        assert mcu.is_connected()
        mcu.set_connected(False)
        assert not mcu.is_connected()
