import csv
import logging
import os
import re
import threading

import pytest

from utils.logger import (
    DEFAULT_FLIGHT_FIELDS,
    ColoredFormatter,
    FlightRecorder,
    setup_logger,
    _BOOT_TIMESTAMP,
)


# =====================================================================
#  Fixtures
# =====================================================================

@pytest.fixture
def log_dir(tmp_path):
    """返回一个临时日志目录路径。"""
    d = tmp_path / "test_logs"
    d.mkdir()
    return str(d)


@pytest.fixture
def fresh_logger_name():
    """
    生成一个唯一的 logger 名称，避免跨用例 handler 污染。
    使用 threading.current_thread 的 ident 与递增计数器组合。
    """
    fresh_logger_name._counter = getattr(fresh_logger_name, "_counter", 0) + 1
    return f"test_logger_{fresh_logger_name._counter}"


# =====================================================================
#  T1–T9：setup_logger 测试
# =====================================================================

class TestT1CreateLoggerInstance:
    """T1 — 创建 Logger 实例，返回 logging.Logger，名称正确。"""

    def test_returns_logger(self, fresh_logger_name, log_dir):
        logger = setup_logger(fresh_logger_name, log_dir=log_dir)
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self, fresh_logger_name, log_dir):
        logger = setup_logger(fresh_logger_name, log_dir=log_dir)
        assert logger.name == fresh_logger_name


class TestT2LogLevel:
    """T2 — 日志级别设置：level="DEBUG" 时 logger.level == DEBUG。"""

    def test_debug_level(self, fresh_logger_name, log_dir):
        logger = setup_logger(fresh_logger_name, level="DEBUG", log_dir=log_dir)
        assert logger.level == logging.DEBUG

    def test_warning_level(self, fresh_logger_name, log_dir):
        # 使用不同名称以避免 handler 复用
        name = fresh_logger_name + "_warn"
        logger = setup_logger(name, level="WARNING", log_dir=log_dir)
        assert logger.level == logging.WARNING


class TestT3ConsoleHandler:
    """T3 — handlers 中包含 StreamHandler。"""

    def test_has_stream_handler(self, fresh_logger_name, log_dir):
        logger = setup_logger(fresh_logger_name, log_dir=log_dir)
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1


class TestT4FileHandler:
    """T4 — handlers 中包含 FileHandler，文件路径在 log_dir 下。"""

    def test_has_file_handler(self, fresh_logger_name, log_dir):
        logger = setup_logger(fresh_logger_name, log_dir=log_dir)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) >= 1

    def test_file_in_log_dir(self, fresh_logger_name, log_dir):
        logger = setup_logger(fresh_logger_name, log_dir=log_dir)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        for fh in file_handlers:
            assert fh.baseFilename.startswith(os.path.abspath(log_dir))


class TestT5TimestampFilename:
    """T5 — 日志文件以时间戳命名，匹配 YYYYMMDD_HHMMSS.log 格式。"""

    def test_filename_pattern(self, fresh_logger_name, log_dir):
        setup_logger(fresh_logger_name, log_dir=log_dir)
        files = os.listdir(log_dir)
        log_files = [f for f in files if f.endswith(".log")]
        assert len(log_files) >= 1
        for f in log_files:
            assert re.match(r"\d{8}_\d{6}\.log$", f), f"文件名不匹配时间戳格式: {f}"


class TestT6AutoCreateLogDir:
    """T6 — log_dir 不存在时自动创建。"""

    def test_auto_create(self, tmp_path, fresh_logger_name):
        new_dir = str(tmp_path / "nonexistent_subdir" / "logs")
        assert not os.path.exists(new_dir)
        setup_logger(fresh_logger_name, log_dir=new_dir)
        assert os.path.isdir(new_dir)


class TestT7NoDuplicateHandlers:
    """T7 — 多次调用 setup_logger 同一 name，Handler 数量不增长。"""

    def test_no_duplicate(self, fresh_logger_name, log_dir):
        logger1 = setup_logger(fresh_logger_name, log_dir=log_dir)
        count1 = len(logger1.handlers)
        logger2 = setup_logger(fresh_logger_name, log_dir=log_dir)
        count2 = len(logger2.handlers)
        assert logger1 is logger2
        assert count1 == count2


class TestT8AllLevelsOutput:
    """T8 — DEBUG/INFO/WARNING/ERROR/CRITICAL 各级别日志均可输出，不抛异常。"""

    def test_all_levels(self, fresh_logger_name, log_dir):
        logger = setup_logger(fresh_logger_name, level="DEBUG", log_dir=log_dir)
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warning msg")
        logger.error("error msg")
        logger.critical("critical msg")


class TestT9FileContainsMessage:
    """T9 — 调用 logger.info("test") 后日志文件中包含 "test"。"""

    def test_file_write(self, fresh_logger_name, log_dir):
        logger = setup_logger(fresh_logger_name, level="DEBUG", log_dir=log_dir)
        unique_msg = "unique_test_message_12345"
        logger.info(unique_msg)

        # 刷新所有 file handler
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.flush()

        # 读取日志文件内容
        log_files = [
            os.path.join(log_dir, f) for f in os.listdir(log_dir)
            if f.endswith(".log")
        ]
        found = False
        for lf in log_files:
            with open(lf, "r", encoding="utf-8") as fh:
                content = fh.read()
                if unique_msg in content:
                    found = True
                    break
        assert found, f"日志文件中未找到消息 '{unique_msg}'"


# =====================================================================
#  T10–T12：ColoredFormatter 测试
# =====================================================================

class TestT10InfoGreenAnsi:
    """T10 — INFO 级别带绿色 ANSI 码。"""

    def test_info_color(self):
        fmt = ColoredFormatter("[%(levelname)s] %(message)s")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        output = fmt.format(record)
        assert "\033[32m" in output


class TestT11WarningYellowAnsi:
    """T11 — WARNING 级别带黄色 ANSI 码。"""

    def test_warning_color(self):
        fmt = ColoredFormatter("[%(levelname)s] %(message)s")
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0,
            msg="caution", args=(), exc_info=None,
        )
        output = fmt.format(record)
        assert "\033[33m" in output


class TestT12ResetAnsi:
    """T12 — 输出末尾包含 RESET 码。"""

    def test_reset_at_end(self):
        fmt = ColoredFormatter("[%(levelname)s] %(message)s")
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="error!", args=(), exc_info=None,
        )
        output = fmt.format(record)
        assert output.endswith("\033[0m")


# =====================================================================
#  T13–T25：FlightRecorder 测试
# =====================================================================

class TestT13InitCreatesCSV:
    """T13 — 初始化创建 CSV 文件，首行为 'timestamp,' + 字段名。"""

    def test_csv_created_with_header(self, log_dir):
        recorder = FlightRecorder(log_dir=log_dir)
        assert os.path.isfile(recorder.filepath)

        with open(recorder.filepath, "r", encoding="utf-8") as f:
            header = f.readline().strip()
        expected = "timestamp," + ",".join(DEFAULT_FLIGHT_FIELDS)
        assert header == expected
        recorder.close()


class TestT14CSVFilenamePattern:
    """T14 — CSV 文件名以 flight_ 前缀 + 时间戳命名。"""

    def test_filename_pattern(self, log_dir):
        recorder = FlightRecorder(log_dir=log_dir)
        basename = os.path.basename(recorder.filepath)
        assert re.match(r"flight_\d{8}_\d{6}\.csv$", basename), \
            f"文件名不匹配模式: {basename}"
        recorder.close()


class TestT15RecordOneLine:
    """T15 — record() 写入一条记录后，CSV 含 2 行（表头 + 数据），timestamp 非空。"""

    def test_one_record(self, log_dir):
        recorder = FlightRecorder(log_dir=log_dir)
        recorder.record({"state": "IDLE", "cmd_vx": 0.0})
        recorder.close()

        with open(recorder.filepath, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["timestamp"] != ""


class TestT16RecordMultipleLines:
    """T16 — record() 多次写入，行数与写入次数 + 1（表头）一致。"""

    def test_multiple_records(self, log_dir):
        n = 5
        recorder = FlightRecorder(log_dir=log_dir)
        for i in range(n):
            recorder.record({"state": f"STATE_{i}"})
        recorder.close()

        with open(recorder.filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 表头 1 行 + n 条数据
        assert len(lines) == n + 1


class TestT17MissingFieldsEmpty:
    """T17 — 传入不完整 data，缺失列为空字符串。"""

    def test_missing_fields(self, log_dir):
        recorder = FlightRecorder(log_dir=log_dir)
        recorder.record({"state": "ALIGN"})  # 只传 state，其余缺失
        recorder.close()

        with open(recorder.filepath, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        row = reader[0]
        assert row["state"] == "ALIGN"
        assert row["cmd_vx"] == ""  # 缺失字段应为空
        assert row["target_u"] == ""


class TestT18ExtraFieldsIgnored:
    """T18 — 传入额外 key，CSV 列数不变。"""

    def test_extra_fields(self, log_dir):
        recorder = FlightRecorder(log_dir=log_dir)
        recorder.record({
            "state": "IDLE",
            "extra_key_1": "should_be_ignored",
            "extra_key_2": 999,
        })
        recorder.close()

        with open(recorder.filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        expected_cols = ["timestamp"] + DEFAULT_FLIGHT_FIELDS
        assert fieldnames == expected_cols


class TestT19CloseRaisesOnRecord:
    """T19 — close() 后 record() 抛出 RuntimeError。"""

    def test_raises_runtime_error(self, log_dir):
        recorder = FlightRecorder(log_dir=log_dir)
        recorder.close()
        with pytest.raises(RuntimeError):
            recorder.record({"state": "IDLE"})


class TestT20ContextManagerAutoClose:
    """T20 — with 退出后文件句柄已关闭。"""

    def test_auto_close(self, log_dir):
        with FlightRecorder(log_dir=log_dir) as recorder:
            recorder.record({"state": "IDLE"})
            filepath = recorder.filepath

        # 退出 with 后应已关闭
        assert recorder._closed is True
        # 文件应存在且可读
        assert os.path.isfile(filepath)


class TestT21CustomFields:
    """T21 — 传入自定义 fields，CSV 表头与之匹配。"""

    def test_custom_fields(self, log_dir):
        custom = ["altitude", "battery_pct", "gps_lat", "gps_lon"]
        recorder = FlightRecorder(log_dir=log_dir, fields=custom)
        recorder.record({"altitude": 10.5, "battery_pct": 85})
        recorder.close()

        with open(recorder.filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        assert fieldnames == ["timestamp"] + custom


class TestT22FilepathProperty:
    """T22 — filepath 属性返回正确路径，与实际创建的文件路径一致。"""

    def test_filepath_matches(self, log_dir):
        recorder = FlightRecorder(log_dir=log_dir)
        fp = recorder.filepath
        assert os.path.isabs(fp)
        assert os.path.isfile(fp)
        recorder.close()


class TestT23AutoCreateLogDir:
    """T23 — log_dir 不存在时自动创建。"""

    def test_auto_create(self, tmp_path):
        new_dir = str(tmp_path / "deep" / "nested" / "flight_logs")
        assert not os.path.exists(new_dir)
        recorder = FlightRecorder(log_dir=new_dir)
        assert os.path.isdir(new_dir)
        recorder.close()


class TestT24CSVReadBack:
    """T24 — 写入数据后用 csv.DictReader 回读验证结构正确。"""

    def test_readback(self, log_dir):
        recorder = FlightRecorder(log_dir=log_dir)
        test_data = {
            "state": "ALIGN",
            "target_u": "312.5",
            "target_v": "248.1",
            "target_theta": "0.12",
            "target_conf": "0.87",
            "cmd_vx": "0.05",
            "cmd_vy": "-0.02",
            "cmd_vyaw": "0.08",
            "dt": "0.067",
        }
        recorder.record(test_data)
        recorder.close()

        with open(recorder.filepath, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        for key, val in test_data.items():
            assert row[key] == str(val), f"字段 {key} 不匹配: {row[key]} != {val}"


class TestT25ThreadSafety:
    """T25 — 多线程并发 record() 不丢行。"""

    def test_concurrent_writes(self, log_dir):
        n_threads = 10
        n_records_per_thread = 50
        total = n_threads * n_records_per_thread

        recorder = FlightRecorder(log_dir=log_dir)

        def worker(thread_id):
            for i in range(n_records_per_thread):
                recorder.record({"state": f"T{thread_id}_R{i}"})

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        recorder.close()

        with open(recorder.filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 表头 1 行 + total 条数据
        assert len(lines) == total + 1, \
            f"预期 {total + 1} 行，实际 {len(lines)} 行（含表头）"
