"""
低延迟视觉流采集模块（M1）。
"""

from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from core.interfaces import IStreamer
from utils.logger import setup_logger


class ZeroLatencyStreamer(IStreamer):
    """
    ESP32-S3 摄像头低延迟采集器。

    线程模型：
      - 后台 daemon 线程独占 VideoCapture（grab + retrieve + 写缓冲区）
      - 主线程 get_latest_frame() 只通过 _lock 读取缓冲区（无 I/O）

    默认行为：
      1. 优先连接 MJPEG 流 `http://<host>:81/stream`
      2. 流不可用时回退轮询 `http://<host>/capture`
      3. `get_latest_frame()` 只返回新鲜帧，过旧帧返回 None
    """

    MODE_CLOSED = "closed"
    MODE_STREAM = "stream"
    MODE_CAPTURE = "capture"

    DEFAULT_HOST = "192.168.43.192"
    DEFAULT_STREAM_PORT = 81

    def __init__(
        self,
        stream_url: str | None = None,
        capture_url: str | None = None,
        host: str | None = None,
        timeout_ms: int = 3000,
        prefer_stream: bool = True,
        capture_poll_interval_s: float = 1.0 / 15.0,
        max_stale_s: float = 2.0,
        stream_retry_interval_s: float = 5.0,
        stream_fail_threshold: int = 5,
        stream_preflight: bool = False,
        auto_start: bool = True,
    ) -> None:
        """
        Args:
            stream_url: MJPEG 流 URL
            capture_url: JPEG 抓拍 URL
            host: ESP32-S3 IP 或 host，未显式传 URL 时用于自动推导端点
            timeout_ms: HTTP 连接/读取超时，ms
            prefer_stream: 是否优先使用 /stream
            capture_poll_interval_s: /capture 回退模式的轮询间隔，s
            max_stale_s: 最新帧超过该时长后视为不可用，s
            stream_retry_interval_s: fallback 模式下重试 /stream 的间隔，s
            stream_fail_threshold: 连续 grab 失败多少次后切换 fallback
            stream_preflight: 打开 VideoCapture 前先用 HTTP 请求检查 /stream 可达性
            auto_start: 是否在构造时立即启动后台采集线程
        """
        if timeout_ms <= 0:
            raise ValueError("timeout_ms 必须 > 0")
        if capture_poll_interval_s <= 0:
            raise ValueError("capture_poll_interval_s 必须 > 0")
        if max_stale_s <= 0:
            raise ValueError("max_stale_s 必须 > 0")
        if stream_retry_interval_s <= 0:
            raise ValueError("stream_retry_interval_s 必须 > 0")
        if stream_fail_threshold <= 0:
            raise ValueError("stream_fail_threshold 必须 > 0")

        host = host or self.DEFAULT_HOST
        self.stream_url = stream_url or self._derive_stream_url(host)
        self.capture_url = capture_url or self._derive_capture_url(host)
        self.timeout_ms = timeout_ms
        self.timeout_s = timeout_ms / 1000.0
        self.prefer_stream = prefer_stream
        self.capture_poll_interval_s = capture_poll_interval_s
        self.max_stale_s = max_stale_s
        self.stream_retry_interval_s = stream_retry_interval_s
        self.stream_fail_threshold = stream_fail_threshold
        self.stream_preflight = stream_preflight

        self._logger = setup_logger("M1.Streamer", level="INFO")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        self._cap: cv2.VideoCapture | None = None
        self._session: requests.Session | None = requests.Session()

        # 以下字段由 _lock 保护
        self._latest_frame: np.ndarray | None = None
        self._last_frame_time: float = 0.0
        self._frame_update_count: int = 0
        self._opened: bool = False
        self._active_mode: str = self.MODE_CLOSED
        self._released: bool = False

        if auto_start:
            self.start()

    @staticmethod
    def _split_host(host: str) -> tuple[str, str]:
        raw = host.strip().rstrip("/")
        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
        scheme = parsed.scheme or "http"
        netloc = parsed.netloc or parsed.path
        if not netloc:
            raise ValueError(f"无效 host: {host}")
        return scheme, netloc

    @classmethod
    def _derive_stream_url(cls, host: str) -> str:
        scheme, netloc = cls._split_host(host)
        if ":" not in netloc:
            netloc = f"{netloc}:{cls.DEFAULT_STREAM_PORT}"
        return f"{scheme}://{netloc}/stream"

    @classmethod
    def _derive_capture_url(cls, host: str) -> str:
        scheme, netloc = cls._split_host(host)
        return f"{scheme}://{netloc}/capture"

    @property
    def current_mode(self) -> str:
        """返回当前采集模式：stream / capture / closed。"""
        with self._lock:
            return self._active_mode

    @property
    def frame_update_count(self) -> int:
        """Return how many source frames have been written into the cache."""
        with self._lock:
            return self._frame_update_count

    def start(self) -> None:
        """启动后台采集线程。重复调用是安全的。release 后不可再调用。"""
        if self._released:
            raise RuntimeError("ZeroLatencyStreamer 已 release，不能再次 start")
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="ZeroLatencyStreamer",
            daemon=True,
        )
        self._worker.start()

    def get_latest_frame(self) -> np.ndarray | None:
        """
        获取最新可用 BGR 帧。

        主线程调用此方法只读取缓冲区（无 I/O）。

        Returns:
            np.ndarray copy 或 None。None 表示未就绪、断连或帧已过期。
        """
        now = time.monotonic()

        with self._lock:
            if self._stop_event.is_set():
                return None

            if self._latest_frame is None:
                return None

            if now - self._last_frame_time > self.max_stale_s:
                return None

            return self._latest_frame.copy()

    def release(self) -> None:
        """停止后台线程并释放 VideoCapture / HTTP session。重复调用安全。"""
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)

        with self._lock:
            self._opened = False
            self._active_mode = self.MODE_CLOSED
            self._latest_frame = None
            self._last_frame_time = 0.0
            self._released = True

        # _cap 由后台线程独占，线程已 join，可安全释放
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._session is not None:
            self._session.close()
            self._session = None

    def is_opened(self) -> bool:
        """检查采集器当前是否有可用采集路径。"""
        with self._lock:
            return not self._stop_event.is_set() and self._opened

    def _worker_loop(self) -> None:
        next_stream_retry = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            can_try_stream = (
                self.prefer_stream
                and bool(self.stream_url)
                and now >= next_stream_retry
            )

            if can_try_stream:
                if self._open_stream_capture():
                    self._run_stream_loop()
                    next_stream_retry = time.monotonic() + self.stream_retry_interval_s
                    continue
                next_stream_retry = now + self.stream_retry_interval_s

            if self.capture_url:
                self._capture_once()
                self._sleep_interruptible(self.capture_poll_interval_s)
            else:
                with self._lock:
                    self._opened = False
                    self._active_mode = self.MODE_CLOSED
                self._sleep_interruptible(0.2)

    def _open_stream_capture(self) -> bool:
        """尝试打开 MJPEG 流。由后台线程调用。"""
        if self.stream_preflight and not self._stream_endpoint_reachable():
            return False

        cap = cv2.VideoCapture(self.stream_url)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            cap.release()
            return False

        # 释放旧的 VideoCapture
        if self._cap is not None:
            self._cap.release()
        self._cap = cap

        with self._lock:
            self._opened = True
            self._active_mode = self.MODE_STREAM

        self._logger.info(f"ZeroLatencyStreamer: 使用 MJPEG 流 {self.stream_url}")
        return True

    def _stream_endpoint_reachable(self) -> bool:
        """
        用 requests 预检 /stream 是否可达。

        OpenCV 的 VideoCapture(url) 可能在无路由/防火墙场景下阻塞较久；预检失败时
        直接回退 /capture，可以让 live test 更快暴露 WSL2 网络路径问题。
        """
        try:
            resp = self._session.get(
                self.stream_url,
                stream=True,
                timeout=self.timeout_s,
            )
            try:
                resp.raise_for_status()
            finally:
                resp.close()
            return True
        except requests.exceptions.RequestException as exc:
            self._logger.debug(f"/stream 预检失败: {exc}")
            return False

    def _run_stream_loop(self) -> None:
        """
        MJPEG 流采集循环。由后台线程调用。

        后台线程独占 VideoCapture：执行 grab() + retrieve() 后将帧
        写入锁保护的缓冲区。主线程 get_latest_frame() 只读取缓冲区。
        """
        failures = 0
        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None:
                return

            # grab() 在锁外执行：网络 I/O 不阻塞主线程
            ok = cap.grab()

            if ok:
                # retrieve() 也在锁外执行：JPEG 解码不阻塞主线程
                ret, frame = cap.retrieve()
                if ret and frame is not None:
                    now = time.monotonic()
                    with self._lock:
                        self._latest_frame = frame
                        self._last_frame_time = now
                        self._frame_update_count += 1
                failures = 0
                time.sleep(0.001)
                continue

            failures += 1
            if failures >= self.stream_fail_threshold:
                self._logger.warning("MJPEG 流连续 grab 失败，切换到 /capture fallback")
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                with self._lock:
                    self._opened = False
                    self._active_mode = self.MODE_CLOSED
                return
            self._sleep_interruptible(0.02)

    def _capture_once(self) -> bool:
        """执行一次 /capture 抓拍。由后台线程调用。"""
        frame = self._fetch_capture_frame()
        if frame is None:
            with self._lock:
                if self._active_mode != self.MODE_STREAM:
                    self._opened = False
                    self._active_mode = self.MODE_CLOSED
            return False

        with self._lock:
            self._latest_frame = frame
            self._last_frame_time = time.monotonic()
            self._frame_update_count += 1
            if self._active_mode != self.MODE_STREAM:
                self._opened = True
                self._active_mode = self.MODE_CAPTURE
        return True

    def _fetch_capture_frame(self) -> np.ndarray | None:
        """HTTP GET /capture 并解码 JPEG。由后台线程调用。"""
        try:
            resp = self._session.get(self.capture_url, timeout=self.timeout_s)
            resp.raise_for_status()
            img_array = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                self._logger.warning("JPEG 解码失败：cv2.imdecode 返回 None")
            return frame
        except requests.exceptions.RequestException as exc:
            self._logger.debug(f"/capture 请求失败: {exc}")
            return None

    def _sleep_interruptible(self, duration_s: float) -> None:
        """可中断的 sleep，响应 _stop_event。"""
        self._stop_event.wait(timeout=duration_s)

    def __enter__(self) -> "ZeroLatencyStreamer":
        return self

    def __exit__(self, *args) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
