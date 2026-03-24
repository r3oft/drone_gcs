import copy
import warnings
from typing import Any

import yaml


class ConfigManager:
    """
    全局配置管理器。

    从 YAML 文件加载参数树，提供点号嵌套访问、命令行覆盖和热重载能力。

    线程安全性：本模块为只读查询（get / to_dict），在主线程初始化阶段完成加载和覆盖。
    多线程运行期间不应调用 reload() 或 override_from_args()。
    """

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        """
        加载 YAML 配置文件，构建参数树。

        Args:
            config_path: YAML 配置文件路径（支持相对路径和绝对路径）。

        Raises:
            FileNotFoundError: 文件不存在时抛出，附带清晰错误信息。
            yaml.YAMLError: YAML 语法错误时抛出，附带行号信息。
        """
        self._config_path: str = config_path
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        """从 self._config_path 加载 YAML 文件到 self._data。"""
        import os
        if not os.path.isfile(self._config_path):
            raise FileNotFoundError(
                f"配置文件不存在：{self._config_path!r}，"
                f"请确认路径是否正确。"
            )

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(
                f"配置文件 YAML 语法错误：{self._config_path!r}\n{exc}"
            ) from exc

        if parsed is None:
            warnings.warn(
                f"配置文件为空：{self._config_path!r}，将使用空配置字典。",
                stacklevel=2,
            )
            self._data = {}
        else:
            self._data = parsed

    def get(self, key: str, default: Any = None) -> Any:
        """
        点号分隔的嵌套 key 读取。

        将 ``"servo.kp.x"`` 按 ``.`` 拆分为 ``["servo", "kp", "x"]``，
        逐层深入字典查找。任何一层 key 不存在时返回 *default*，不抛出异常。

        Args:
            key:     点号分隔的参数路径，如 ``"servo.kp.x"``。
            default: key 不存在时的返回值（默认 ``None``）。

        Returns:
            对应的参数值，或者 *default*。
        """
        keys = key.split(".")
        node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def override_from_args(self, cli_args: dict) -> None:
        """
        用命令行参数字典覆盖已加载的配置值。

        键格式与 :meth:`get` 相同（点号分隔）。如果中间层不存在，会自动创建空字典。

        覆盖优先级：命令行参数 > YAML 文件默认值。

        Args:
            cli_args: 形如 ``{"servo.kp.x": 0.005, "fsm.tick_rate_hz": 10}``
                      的扁平化键值对字典。
        """
        for dotted_key, value in cli_args.items():
            keys = dotted_key.split(".")
            node = self._data
            for k in keys[:-1]:  
                if k not in node or not isinstance(node[k], dict):
                    node[k] = {} 
                node = node[k]
            node[keys[-1]] = value  

    def reload(self) -> None:
        """
        从文件重新加载配置（热重载）。

        .. warning::
            热重载会**清除**之前通过 :meth:`override_from_args` 注入的覆盖值。
            如果需要保留覆盖值，请在调用 ``reload()`` 后重新调用 ``override_from_args()``。
        """
        self._load()

    def to_dict(self) -> dict:
        """
        返回当前完整配置树的深拷贝字典。

        Returns:
            当前配置树的深拷贝。
        """
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return (
            f"ConfigManager(config_path={self._config_path!r}, "
            f"keys={list(self._data.keys())})"
        )
