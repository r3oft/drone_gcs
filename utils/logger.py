import csv
import logging
import os
import threading
from datetime import datetime

# 模块级启动时间戳（同一次运行中保持一致）
_BOOT_TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")

# 默认黑匣子字段（这个需要根据实际的日志格式做修改）
DEFAULT_FLIGHT_FIELDS: list[str] = [
    "state", "target_u", "target_v", "target_theta", "target_conf",
    "cmd_vx", "cmd_vy", "cmd_vyaw", "dt",
]

class ColoredFormatter(logging.Formatter):
    """
    ANSI 彩色日志格式化器。

    根据日志级别为控制台输出添加颜色前缀与重置后缀，提升终端可读性。
    """

    COLORS: dict[int, str] = {
        logging.DEBUG:    "\033[36m",     # Cyan
        logging.INFO:     "\033[32m",     # Green
        logging.WARNING:  "\033[33m",     # Yellow
        logging.ERROR:    "\033[31m",     # Red
        logging.CRITICAL: "\033[1;31m",   # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录，按级别添加 ANSI 颜色。"""
        color = self.COLORS.get(record.levelno, "")
        formatted = super().format(record)
        return f"{color}{formatted}{self.RESET}"


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = "logs/",
) -> logging.Logger:
    """
    创建并返回一个配置好的 Logger 实例。

    Features:
        - 控制台输出带 ANSI 颜色分级
        - 文件输出以启动时间戳命名，自动归档到 log_dir
        - 防止重复添加 Handler

    Args:
        name:    Logger 名称（推荐使用模块名，如 "M2.perception"）
        level:   日志级别字符串（DEBUG / INFO / WARNING / ERROR / CRITICAL）
        log_dir: 日志文件输出目录

    Returns:
        配置好的 logging.Logger 实例
    """
    logger = logging.getLogger(name)

    # 防止重复添加 Handler
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ── 控制台 Handler ──
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_fmt = ColoredFormatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # 使用 msecs 精确到毫秒：覆写 default_msec_format
    console_fmt.default_msec_format = "%s.%03d"
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # ── 文件 Handler ──
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{_BOOT_TIMESTAMP}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_fmt.default_msec_format = "%s.%03d"
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger

class FlightRecorder:
    """
    飞行数据黑匣子。

    以 CSV 格式逐帧记录关键遥测数据，供事后回放与故障分析。

    线程安全：record() 使用 threading.Lock 保护文件写入。
    """

    def __init__(
        self,
        log_dir: str = "logs/",
        fields: list[str] | None = None,
    ) -> None:
        """
        初始化记录器，创建 CSV 文件并写入表头。

        Args:
            log_dir:  CSV 输出目录（不存在时自动创建）
            fields:   CSV 列名列表（不含 timestamp）。
                      为 None 时使用 DEFAULT_FLIGHT_FIELDS。
        """
        self._fields = list(fields) if fields is not None else list(DEFAULT_FLIGHT_FIELDS)
        self._all_columns = ["timestamp"] + self._fields
        self._lock = threading.Lock()
        self._closed = False

        # 创建输出目录
        os.makedirs(log_dir, exist_ok=True)

        # 生成 CSV 文件路径
        csv_filename = f"flight_{_BOOT_TIMESTAMP}.csv"
        self._filepath = os.path.join(os.path.abspath(log_dir), csv_filename)

        # 打开文件并写入表头
        self._file = open(self._filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._all_columns,
            extrasaction="ignore",
            restval="",
        )
        self._writer.writeheader()
        self._file.flush()

    def record(self, data: dict) -> None:
        """
        写入一条帧数据记录。

        自动附加 ISO 8601 时间戳作为首列。
        data 中缺失的字段填空字符串，多余的字段被忽略。

        Raises:
            RuntimeError: 文件句柄已关闭时抛出
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("FlightRecorder 已关闭，无法继续写入")

            row = dict(data)
            row["timestamp"] = datetime.now().isoformat()
            self._writer.writerow(row)
            self._file.flush()

    def close(self) -> None:
        """刷新缓冲区并关闭文件句柄。"""
        with self._lock:
            if not self._closed:
                self._file.flush()
                self._file.close()
                self._closed = True

    @property
    def filepath(self) -> str:
        """返回当前 CSV 文件的绝对路径。"""
        return self._filepath

    def __enter__(self) -> "FlightRecorder":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        """防止未显式关闭时泄漏文件句柄。"""
        if not self._closed:
            self.close()
