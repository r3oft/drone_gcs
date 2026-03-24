"""
工具模块包（utils）。

导出核心工具类，供上层模块 `from utils import ConfigManager` 使用。
"""

from utils.config_manager import ConfigManager

__all__ = ["ConfigManager"]
