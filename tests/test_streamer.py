import time

import cv2
import numpy as np
import pytest

from core.interfaces import IStreamer
from core.streamer import ZeroLatencyStreamer


def make_frame(value: int = 96) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


def make_jpeg_bytes(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class DummyResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class DummySession:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.urls: list[str] = []
        self.closed = False

    def get(self, url: str, timeout: float):
        self.urls.append(url)
        return DummyResponse(self.content)

    def close(self) -> None:
        self.closed = True


class FakeVideoCapture:
    opened = True
    frame = make_frame(128)
    instances = []

    def __init__(self, url: str) -> None:
        self.url = url
        self.released = False
        self.grab_count = 0
        FakeVideoCapture.instances.append(self)

    def set(self, prop_id, value) -> bool:
        return True

    def isOpened(self) -> bool:
        return FakeVideoCapture.opened and not self.released

    def grab(self) -> bool:
        if not self.isOpened():
            return False
        self.grab_count += 1
        return True

    def retrieve(self):
        if not self.isOpened():
            return False, None
        return True, FakeVideoCapture.frame.copy()

    def release(self) -> None:
        self.released = True


@pytest.fixture(autouse=True)
def reset_fake_capture():
    FakeVideoCapture.opened = True
    FakeVideoCapture.frame = make_frame(128)
    FakeVideoCapture.instances = []
    yield


# ═══════════════════════════════════════════════════════
#  URL 推导测试
# ═══════════════════════════════════════════════════════


class TestZeroLatencyStreamerUrls:
    def test_host_derives_stream_and_capture_urls(self):
        streamer = ZeroLatencyStreamer(host="192.168.43.192", auto_start=False)

        assert streamer.stream_url == "http://192.168.43.192:81/stream"
        assert streamer.capture_url == "http://192.168.43.192/capture"
        assert isinstance(streamer, IStreamer)
        streamer.release()

    def test_explicit_urls_override_host(self):
        streamer = ZeroLatencyStreamer(
            host="192.168.43.192",
            stream_url="http://camera.local:8080/video",
            capture_url="http://camera.local/snap",
            auto_start=False,
        )

        assert streamer.stream_url == "http://camera.local:8080/video"
        assert streamer.capture_url == "http://camera.local/snap"
        streamer.release()


# ═══════════════════════════════════════════════════════
#  Capture 模式测试
# ═══════════════════════════════════════════════════════


class TestZeroLatencyStreamerCapture:
    def test_capture_jpeg_decodes_to_bgr_frame(self):
        original = make_frame(72)
        session = DummySession(make_jpeg_bytes(original))
        streamer = ZeroLatencyStreamer(
            host="192.168.43.192",
            prefer_stream=False,
            auto_start=False,
        )
        streamer._session = session

        assert streamer._capture_once() is True
        frame = streamer.get_latest_frame()

        assert frame is not None
        assert frame.shape == original.shape
        assert frame.dtype == np.uint8
        assert streamer.frame_update_count == 1
        assert session.urls == ["http://192.168.43.192/capture"]
        streamer.release()

    def test_capture_fallback_used_when_stream_unavailable(self, monkeypatch):
        FakeVideoCapture.opened = False
        capture_frame = make_frame(44)
        session = DummySession(make_jpeg_bytes(capture_frame))

        monkeypatch.setattr("core.streamer.cv2.VideoCapture", FakeVideoCapture)
        monkeypatch.setattr("core.streamer.requests.Session", lambda: session)

        streamer = ZeroLatencyStreamer(
            host="192.168.43.192",
            capture_poll_interval_s=0.01,
            stream_retry_interval_s=0.2,
        )
        try:
            assert wait_until(lambda: streamer.get_latest_frame() is not None)
            assert streamer.current_mode == ZeroLatencyStreamer.MODE_CAPTURE
            assert streamer.is_opened()
        finally:
            streamer.release()


# ═══════════════════════════════════════════════════════
#  Stream 模式测试
# ═══════════════════════════════════════════════════════


class TestZeroLatencyStreamerStream:
    def test_stream_mode_is_preferred_when_available(self, monkeypatch):
        monkeypatch.setattr("core.streamer.cv2.VideoCapture", FakeVideoCapture)

        streamer = ZeroLatencyStreamer(
            host="192.168.43.192",
            capture_poll_interval_s=0.01,
        )
        try:
            assert wait_until(
                lambda: (
                    streamer.current_mode == ZeroLatencyStreamer.MODE_STREAM
                    and FakeVideoCapture.instances
                    and FakeVideoCapture.instances[0].grab_count > 0
                )
            )
            frame = streamer.get_latest_frame()

            assert frame is not None
            assert frame.shape == FakeVideoCapture.frame.shape
            assert streamer.frame_update_count > 0
            assert FakeVideoCapture.instances[0].url == "http://192.168.43.192:81/stream"
            assert FakeVideoCapture.instances[0].grab_count > 0
        finally:
            streamer.release()

    def test_stream_grab_fail_triggers_fallback(self, monkeypatch):
        """T1: stream 中途断开后自动切换 capture (B1 fix 验证)。"""

        class BreakableVideoCapture(FakeVideoCapture):
            """grab() 在达到计数后开始持续失败。"""
            _break_after = 3

            def grab(self) -> bool:
                if not self.isOpened():
                    return False
                self.grab_count += 1
                return self.grab_count <= BreakableVideoCapture._break_after

        capture_frame = make_frame(77)
        session = DummySession(make_jpeg_bytes(capture_frame))

        monkeypatch.setattr("core.streamer.cv2.VideoCapture", BreakableVideoCapture)
        monkeypatch.setattr("core.streamer.requests.Session", lambda: session)

        streamer = ZeroLatencyStreamer(
            host="192.168.43.192",
            stream_fail_threshold=3,
            capture_poll_interval_s=0.01,
            stream_retry_interval_s=10.0,  # 避免在测试期间重试 stream
        )
        try:
            # 等待从 stream 切换到 capture
            assert wait_until(
                lambda: streamer.current_mode in (
                    ZeroLatencyStreamer.MODE_CAPTURE,
                    ZeroLatencyStreamer.MODE_CLOSED,
                ),
                timeout_s=2.0,
            )
            # 等待 capture 帧到达
            assert wait_until(
                lambda: streamer.get_latest_frame() is not None,
                timeout_s=2.0,
            )
        finally:
            streamer.release()


# ═══════════════════════════════════════════════════════
#  帧有效性测试
# ═══════════════════════════════════════════════════════


class TestZeroLatencyStreamerFrameValidity:
    def test_no_frame_returns_none(self):
        streamer = ZeroLatencyStreamer(auto_start=False)

        assert streamer.get_latest_frame() is None
        streamer.release()

    def test_stale_frame_returns_none(self):
        streamer = ZeroLatencyStreamer(auto_start=False, max_stale_s=0.1)
        with streamer._lock:
            streamer._latest_frame = make_frame(10)
            streamer._last_frame_time = time.monotonic() - 1.0
            streamer._opened = True

        assert streamer.get_latest_frame() is None
        streamer.release()

    def test_release_then_get_frame_returns_none(self):
        streamer = ZeroLatencyStreamer(auto_start=False)
        with streamer._lock:
            streamer._latest_frame = make_frame(10)
            streamer._last_frame_time = time.monotonic()
            streamer._opened = True

        streamer.release()

        assert streamer.get_latest_frame() is None
        assert not streamer.is_opened()

    def test_release_is_idempotent(self):
        streamer = ZeroLatencyStreamer(auto_start=False)

        streamer.release()
        streamer.release()

        assert not streamer.is_opened()

    def test_get_latest_frame_returns_copy(self):
        streamer = ZeroLatencyStreamer(auto_start=False)
        with streamer._lock:
            streamer._latest_frame = make_frame(33)
            streamer._last_frame_time = time.monotonic()
            streamer._opened = True

        first = streamer.get_latest_frame()
        assert first is not None
        first[:, :] = 255
        second = streamer.get_latest_frame()

        assert second is not None
        assert not np.array_equal(first, second)
        streamer.release()


# ═══════════════════════════════════════════════════════
#  B3 修复验证：release 后 start 应抛异常
# ═══════════════════════════════════════════════════════


class TestZeroLatencyStreamerLifecycle:
    def test_start_after_release_raises(self):
        """B3: release 后不可再 start。"""
        streamer = ZeroLatencyStreamer(auto_start=False)
        streamer.release()

        with pytest.raises(RuntimeError, match="已 release"):
            streamer.start()

    def test_context_manager(self):
        """确认 with 语句正确释放资源。"""
        with ZeroLatencyStreamer(auto_start=False) as s:
            assert not s.is_opened()
        # 退出 with 后已 release
        assert s._released is True


# ═══════════════════════════════════════════════════════
#  B4 修复验证：capture 失败时模式应为 CLOSED
# ═══════════════════════════════════════════════════════


class TestZeroLatencyStreamerCaptureFailure:
    def test_capture_fail_sets_mode_closed(self):
        """B4: capture 失败时 mode 应为 CLOSED 而非矛盾的 CAPTURE。"""
        import requests as req

        streamer = ZeroLatencyStreamer(
            host="192.168.43.192",
            prefer_stream=False,
            auto_start=False,
        )

        # 注入一个会失败的 session
        class FailSession:
            def get(self, url, timeout):
                raise req.exceptions.ConnectionError("mock fail")

            def close(self):
                pass

        streamer._session = FailSession()
        streamer._capture_once()

        assert streamer.current_mode == ZeroLatencyStreamer.MODE_CLOSED
        assert not streamer.is_opened()
        streamer.release()


# ═══════════════════════════════════════════════════════
#  R2 修复验证：VideoCapture 超时设置
# ═══════════════════════════════════════════════════════


class TestZeroLatencyStreamerTimeout:
    def test_timeout_ms_is_stored(self):
        """R2: 确认 timeout_ms 被保存供 VideoCapture 使用。"""
        streamer = ZeroLatencyStreamer(
            host="192.168.43.192",
            timeout_ms=5000,
            auto_start=False,
        )
        assert streamer.timeout_ms == 5000
        assert streamer.timeout_s == 5.0
        streamer.release()
