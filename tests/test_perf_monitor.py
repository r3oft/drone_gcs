import logging
import threading
import time
from io import StringIO
from unittest.mock import MagicMock

import pytest

from utils.perf_monitor import PerfMonitor


# =====================================================================
#  T1–T3：初始化与基本属性测试
# =====================================================================

class TestT1DefaultWindowSize:
    """T1 — 默认窗口大小 == 30。"""

    def test_default(self):
        perf = PerfMonitor()
        assert perf.window_size == 30


class TestT2CustomWindowSize:
    """T2 — 自定义窗口大小。"""

    def test_custom(self):
        perf = PerfMonitor(window_size=10)
        assert perf.window_size == 10


class TestT3InitialStatsEmpty:
    """T3 — 无任何 measure 调用时 get_stats 返回空 dict。"""

    def test_empty_stats(self):
        perf = PerfMonitor()
        assert perf.get_stats() == {}


# =====================================================================
#  T4–T8：measure() 上下文管理器测试
# =====================================================================

class TestT4MeasureCreatesLabel:
    """T4 — measure 后 get_stats 中该 label 存在。"""

    def test_label_exists(self):
        perf = PerfMonitor()
        with perf.measure("test"):
            pass
        stats = perf.get_stats()
        assert "test" in stats


class TestT5TimingReasonable:
    """T5 — time.sleep(0.05) 后 avg_ms 在合理范围内。"""

    def test_sleep_timing(self):
        perf = PerfMonitor()
        with perf.measure("sleep_test"):
            time.sleep(0.05)
        stats = perf.get_stats()
        avg = stats["sleep_test"]["avg_ms"]
        # sleep(50ms) 应在 45~80ms 之间（考虑系统调度误差）
        assert 40 <= avg <= 100, f"avg_ms={avg} 不在合理范围内"


class TestT6MultipleMeasuresSameLabel:
    """T6 — 多次测量同一 label，count 递增。"""

    def test_count_increments(self):
        perf = PerfMonitor()
        n = 5
        for _ in range(n):
            with perf.measure("repeat"):
                pass
        stats = perf.get_stats()
        assert stats["repeat"]["count"] == n


class TestT7DifferentLabels:
    """T7 — 测量不同 label，各自独立。"""

    def test_independent_labels(self):
        perf = PerfMonitor()
        with perf.measure("alpha"):
            pass
        with perf.measure("beta"):
            pass
        stats = perf.get_stats()
        assert "alpha" in stats
        assert "beta" in stats
        assert len(stats) == 2


class TestT8ExceptionSafety:
    """T8 — 被测代码抛异常后，耗时仍被正确记录。"""

    def test_exception_recorded(self):
        perf = PerfMonitor()
        with pytest.raises(ValueError):
            with perf.measure("error_block"):
                raise ValueError("test error")
        stats = perf.get_stats()
        assert "error_block" in stats
        assert stats["error_block"]["count"] == 1


# =====================================================================
#  T9–T11：滑动窗口测试
# =====================================================================

class TestT9WindowCapacity:
    """T9 — 窗口容量限制：写入超过 window_size 条后，窗口内样本数不超过 window_size。"""

    def test_capacity(self):
        ws = 5
        perf = PerfMonitor(window_size=ws)
        for _ in range(ws + 10):
            with perf.measure("cap_test"):
                pass
        # 内部窗口验证：通过 get_stats 间接验证（count 应为 ws+10）
        stats = perf.get_stats()
        assert stats["cap_test"]["count"] == ws + 10
        # 窗口内样本数通过内部 deque 验证
        assert len(perf._windows["cap_test"]) == ws


class TestT10WindowEvictsOld:
    """T10 — 先写入慢样本、再写入快样本，avg_ms 应下降。"""

    def test_eviction(self):
        perf = PerfMonitor(window_size=3)
        # 手动注入已知耗时
        perf._record("evict", 100.0)
        perf._record("evict", 100.0)
        perf._record("evict", 100.0)
        avg_before = perf.get_stats()["evict"]["avg_ms"]

        # 写入快样本，淘汰慢样本
        perf._record("evict", 1.0)
        perf._record("evict", 1.0)
        perf._record("evict", 1.0)
        avg_after = perf.get_stats()["evict"]["avg_ms"]

        assert avg_after < avg_before


class TestT11CountUnaffectedByWindow:
    """T11 — 总 count 持续累加，不因窗口满而重置。"""

    def test_count_continues(self):
        ws = 3
        perf = PerfMonitor(window_size=ws)
        total = 20
        for _ in range(total):
            perf._record("count_test", 1.0)
        stats = perf.get_stats()
        assert stats["count_test"]["count"] == total


# =====================================================================
#  T12–T16：统计量准确性测试
# =====================================================================

class TestT12AvgCorrect:
    """T12 — avg_ms 计算正确。"""

    def test_avg(self):
        perf = PerfMonitor(window_size=10)
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for v in values:
            perf._record("avg", v)
        stats = perf.get_stats()
        assert stats["avg"]["avg_ms"] == 30.0


class TestT13MaxCorrect:
    """T13 — max_ms 为最大值。"""

    def test_max(self):
        perf = PerfMonitor(window_size=10)
        values = [10.0, 50.0, 20.0, 40.0, 30.0]
        for v in values:
            perf._record("max_test", v)
        stats = perf.get_stats()
        assert stats["max_test"]["max_ms"] == 50.0


class TestT14MinCorrect:
    """T14 — min_ms 为最小值。"""

    def test_min(self):
        perf = PerfMonitor(window_size=10)
        values = [10.0, 50.0, 20.0, 40.0, 30.0]
        for v in values:
            perf._record("min_test", v)
        stats = perf.get_stats()
        assert stats["min_test"]["min_ms"] == 10.0


class TestT15FpsCorrect:
    """T15 — fps == 1000.0 / avg_ms。"""

    def test_fps(self):
        perf = PerfMonitor(window_size=10)
        # avg_ms = 50.0 → fps = 20.0
        for _ in range(5):
            perf._record("fps_test", 50.0)
        stats = perf.get_stats()
        assert stats["fps_test"]["fps"] == 20.0


class TestT16EmptyFpsZero:
    """T16 — 无数据时相关 label 不出现在 stats 中（避免除零）。"""

    def test_no_data_no_entry(self):
        perf = PerfMonitor()
        stats = perf.get_stats()
        # 空 stats 时不会有任何 fps 字段
        assert stats == {}


# =====================================================================
#  T17–T19：print_summary() 测试
# =====================================================================

class TestT17PrintContainsLabels:
    """T17 — 输出包含所有已注册 label。"""

    def test_labels_in_output(self, capsys):
        perf = PerfMonitor()
        perf._record("alpha", 10.0)
        perf._record("beta", 20.0)
        perf.print_summary()
        captured = capsys.readouterr().out
        assert "alpha" in captured
        assert "beta" in captured


class TestT18PrintWithLogger:
    """T18 — 传入 mock logger，验证 logger.info 被调用。"""

    def test_logger_called(self):
        perf = PerfMonitor()
        perf._record("label_x", 10.0)
        mock_logger = MagicMock()
        perf.print_summary(logger=mock_logger)
        assert mock_logger.info.call_count > 0


class TestT19PrintEmptyNoCrash:
    """T19 — 无 measure 调用时 print_summary 不抛异常。"""

    def test_empty_no_crash(self, capsys):
        perf = PerfMonitor()
        perf.print_summary()  # 不应抛异常
        captured = capsys.readouterr().out
        assert "暂无数据" in captured


# =====================================================================
#  T20–T21：reset() 测试
# =====================================================================

class TestT20ResetClearsAll:
    """T20 — reset() 后 get_stats() 返回空 dict。"""

    def test_reset(self):
        perf = PerfMonitor()
        perf._record("data", 10.0)
        assert perf.get_stats() != {}
        perf.reset()
        assert perf.get_stats() == {}


class TestT21ResetThenMeasure:
    """T21 — reset() 后 measure 正常工作，count 从 0 重新计数。"""

    def test_remeasure_after_reset(self):
        perf = PerfMonitor()
        for _ in range(10):
            perf._record("old", 1.0)
        perf.reset()

        perf._record("new", 5.0)
        stats = perf.get_stats()
        assert "old" not in stats
        assert "new" in stats
        assert stats["new"]["count"] == 1


# =====================================================================
#  T22–T23：线程安全测试
# =====================================================================

class TestT22ConcurrentSameLabel:
    """T22 — 多线程并发 measure 同一 label，count 不丢。"""

    def test_concurrent_same(self):
        perf = PerfMonitor(window_size=1000)
        n_threads = 10
        n_per_thread = 100
        total = n_threads * n_per_thread

        def worker():
            for _ in range(n_per_thread):
                with perf.measure("concurrent"):
                    pass

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = perf.get_stats()
        assert stats["concurrent"]["count"] == total


class TestT23ConcurrentDifferentLabels:
    """T23 — 多线程分别 measure 不同 label，各自统计独立正确。"""

    def test_concurrent_different(self):
        perf = PerfMonitor(window_size=500)
        n_per_thread = 50

        def worker(label):
            for _ in range(n_per_thread):
                with perf.measure(label):
                    pass

        labels = [f"thread_{i}" for i in range(8)]
        threads = [threading.Thread(target=worker, args=(l,)) for l in labels]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = perf.get_stats()
        for label in labels:
            assert label in stats
            assert stats[label]["count"] == n_per_thread
