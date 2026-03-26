import time
import threading
from collections import deque
from contextlib import contextmanager


class PerfMonitor:

    def __init__(
        self,
        window_size: int = 30,
        enable: bool = True,
        warn_threshold_ms: float = 80.0,
    ) -> None:
        """
        Args:
            window_size:      滑动窗口容量（样本数），
                              默认 30 — 在 15Hz 主循环下约覆盖 2 秒。
            enable:           全局开关，为 False 时 measure() 不采集数据。
            warn_threshold_ms: 单次耗时告警阈值（ms），超过时返回告警标记。
        """
        self._window_size = window_size
        self._enable = enable
        self._warn_threshold_ms = warn_threshold_ms
        self._windows: dict[str, deque[float]] = {}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def enable(self) -> bool:
        return self._enable

    @enable.setter
    def enable(self, value: bool) -> None:
        self._enable = value

    def _record(self, label: str, elapsed_ms: float) -> None:
        with self._lock:
            if label not in self._windows:
                self._windows[label] = deque(maxlen=self._window_size)
                self._counts[label] = 0
            self._windows[label].append(elapsed_ms)
            self._counts[label] += 1

    @contextmanager
    def measure(self, label: str):
        """
        上下文管理器，测量被包裹代码块的执行耗时。

        用法:
            with perf.measure("yolo_infer"):
                result = model.predict(frame)

        Args:
            label: 标签名称，用于区分不同测量段。

        Yields:
            None
        """
        if not self._enable:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._record(label, elapsed_ms)

    def get_stats(self) -> dict[str, dict]:
        """
        返回所有已注册 label 的滑动窗口统计信息。

        Returns:
            字典，key 为 label，value 为包含以下字段的字典：
            - avg_ms (float): 窗口内平均耗时（毫秒）
            - max_ms (float): 窗口内最大耗时（毫秒）
            - min_ms (float): 窗口内最小耗时（毫秒）
            - fps (float): 等效帧率（1000 / avg_ms）
            - count (int): 累计测量次数
        """
        with self._lock:
            stats: dict[str, dict] = {}
            for label, window in self._windows.items():
                if len(window) == 0:
                    continue
                samples = list(window)
                avg = sum(samples) / len(samples)
                stats[label] = {
                    "avg_ms": round(avg, 2),
                    "max_ms": round(max(samples), 2),
                    "min_ms": round(min(samples), 2),
                    "fps": round(1000.0 / avg, 2) if avg > 0 else 0.0,
                    "count": self._counts[label],
                }
            return stats

    def print_summary(self, logger=None) -> None:
        """
        打印格式化的性能总览表格。

        Args:
            logger: 可选的 logging.Logger 实例。
                    若提供，使用 logger.info() 输出；否则使用 print()。
        """
        stats = self.get_stats()
        if not stats:
            line = "PerfMonitor: 暂无数据"
            if logger:
                logger.info(line)
            else:
                print(line)
            return

        header = (
            "┌─────────────────── PerfMonitor Summary ───────────────────┐\n"
            "│ Label            │ Avg(ms) │ Max(ms) │ Min(ms) │ FPS      │\n"
            "├──────────────────┼─────────┼─────────┼─────────┼──────────┤"
        )
        footer = (
            "└──────────────────┴─────────┴─────────┴─────────┴──────────┘"
        )

        rows = []
        for label, s in stats.items():
            row = (
                f"│ {label:<16s} │ {s['avg_ms']:>7.2f} │ "
                f"{s['max_ms']:>7.2f} │ {s['min_ms']:>7.2f} │ "
                f"{s['fps']:>8.2f} │"
            )
            rows.append(row)

        output = "\n".join([header] + rows + [footer])

        if logger:
            for line in output.split("\n"):
                logger.info(line)
        else:
            print(output)

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()
            self._counts.clear()
