"""
pytest 根配置文件。

存在此文件时，pytest 会自动将项目根目录（drone_gcs/）插入 sys.path，
从而保证 `from utils.config_manager import ConfigManager` 等绝对导入
在任意工作目录下均可正常解析。
"""
