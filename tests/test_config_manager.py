import os
import textwrap

import pytest
import yaml

from utils.config_manager import ConfigManager


# =====================================================================
#  Fixtures：生成临时 YAML 文件
# =====================================================================

SAMPLE_YAML = textwrap.dedent("""\
    stream:
      url: "http://192.168.4.2:81/stream"
      timeout_ms: 3000

    perception:
      weights: "weights/yolov8n-obb-custom.pt"
      conf_threshold: 0.6
      device: "cuda:0"

    servo:
      kp: { x: 0.003, y: 0.003, yaw: 0.6 }
      kd: { x: 0.001, y: 0.001, yaw: 0.1 }
      deadband: { x: 15, y: 15, yaw: 0.05 }
      max_vel: { x: 0.3, y: 0.3, yaw: 0.5 }

    fsm:
      tick_rate_hz: 15
""")


@pytest.fixture
def sample_yaml_path(tmp_path):
    """创建包含典型配置内容的临时 YAML 文件。"""
    p = tmp_path / "sample.yaml"
    p.write_text(SAMPLE_YAML, encoding="utf-8")
    return str(p)


@pytest.fixture
def empty_yaml_path(tmp_path):
    """创建一个空的 YAML 文件。"""
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    return str(p)


@pytest.fixture
def bad_yaml_path(tmp_path):
    """创建一个 YAML 语法错误的文件。"""
    p = tmp_path / "bad.yaml"
    p.write_text("key: [unclosed bracket\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def config(sample_yaml_path):
    """返回已加载合法 YAML 的 ConfigManager 实例。"""
    return ConfigManager(sample_yaml_path)


# =====================================================================
#  T1：加载合法 YAML 文件
# =====================================================================

class TestT1LoadValidYAML:
    """T1 — 加载合法 YAML 文件，self._data 非空，无异常。"""

    def test_data_not_empty(self, config):
        data = config.to_dict()
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_no_exception(self, sample_yaml_path):
        """构造过程不应抛出异常。"""
        cfg = ConfigManager(sample_yaml_path)
        assert cfg is not None


# =====================================================================
#  T2：加载不存在的文件
# =====================================================================

class TestT2FileNotFound:
    """T2 — 加载不存在的文件应抛出 FileNotFoundError。"""

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ConfigManager("/non/existent/path/nothing.yaml")


# =====================================================================
#  T3：加载空 YAML 文件
# =====================================================================

class TestT3EmptyYAML:
    """T3 — 加载空 YAML 文件，self._data 为 {}，不崩溃。"""

    def test_empty_data(self, empty_yaml_path):
        with pytest.warns(UserWarning, match="配置文件为空"):
            cfg = ConfigManager(empty_yaml_path)
        assert cfg.to_dict() == {}


# =====================================================================
#  T4：get("servo.kp.x") 返回 0.003
# =====================================================================

class TestT4GetLeafValue:
    """T4 — get('servo.kp.x') 返回 0.003。"""

    def test_get_leaf(self, config):
        assert config.get("servo.kp.x") == 0.003


# =====================================================================
#  T5：get("servo.kp") 返回子字典
# =====================================================================

class TestT5GetSubDict:
    """T5 — get('servo.kp') 返回完整子字典。"""

    def test_get_sub_dict(self, config):
        expected = {"x": 0.003, "y": 0.003, "yaw": 0.6}
        assert config.get("servo.kp") == expected


# =====================================================================
#  T6：get("nonexistent.key") 返回 None
# =====================================================================

class TestT6GetNonexistentDefault:
    """T6 — get('nonexistent.key') 返回 None。"""

    def test_returns_none(self, config):
        assert config.get("nonexistent.key") is None


# =====================================================================
#  T7：get("nonexistent.key", 42) 返回 42
# =====================================================================

class TestT7GetNonexistentCustomDefault:
    """T7 — get('nonexistent.key', 42) 返回 42。"""

    def test_returns_custom_default(self, config):
        assert config.get("nonexistent.key", 42) == 42


# =====================================================================
#  T8：override_from_args 覆盖后 get 返回新值
# =====================================================================

class TestT8OverrideFromArgs:
    """T8 — override_from_args 后 get 返回覆盖值。"""

    def test_override(self, config):
        config.override_from_args({"servo.kp.x": 0.005})
        assert config.get("servo.kp.x") == 0.005

    def test_other_values_unaffected(self, config):
        """覆盖某个 key 不应影响同级别的其他 key。"""
        config.override_from_args({"servo.kp.x": 0.005})
        assert config.get("servo.kp.y") == 0.003
        assert config.get("servo.kp.yaw") == 0.6


# =====================================================================
#  T9：覆盖不存在的嵌套 key，自动创建中间层
# =====================================================================

class TestT9OverrideNewKey:
    """T9 — 覆盖不存在的嵌套 key 应自动创建中间层并写入。"""

    def test_create_new_path(self, config):
        config.override_from_args({"new.section.param": 1})
        assert config.get("new.section.param") == 1

    def test_intermediate_layers_created(self, config):
        config.override_from_args({"a.b.c.d": "deep"})
        assert config.get("a.b.c.d") == "deep"
        # 中间层应为字典
        assert isinstance(config.get("a.b.c"), dict)
        assert isinstance(config.get("a.b"), dict)
        assert isinstance(config.get("a"), dict)


# =====================================================================
#  T10：reload() 后覆盖值被清除
# =====================================================================

class TestT10Reload:
    """T10 — reload() 后覆盖值被清除，恢复文件原始值。"""

    def test_reload_clears_override(self, config):
        # 先覆盖
        config.override_from_args({"servo.kp.x": 0.005})
        assert config.get("servo.kp.x") == 0.005

        # 热重载
        config.reload()

        # 恢复原始值
        assert config.get("servo.kp.x") == 0.003


# =====================================================================
#  T11：to_dict() 深拷贝验证
# =====================================================================

class TestT11DeepCopy:
    """T11 — to_dict() 返回的字典修改后不影响内部状态。"""

    def test_deep_copy_isolation(self, config):
        exported = config.to_dict()
        # 修改导出的字典
        exported["servo"]["kp"]["x"] = 999.0
        # 内部状态不受影响
        assert config.get("servo.kp.x") == 0.003


# =====================================================================
#  T12：加载含语法错误的 YAML
# =====================================================================

class TestT12BadYAML:
    """T12 — 加载含语法错误的 YAML 应抛出 yaml.YAMLError。"""

    def test_raises_yaml_error(self, bad_yaml_path):
        with pytest.raises(yaml.YAMLError):
            ConfigManager(bad_yaml_path)


# =====================================================================
#  集成测试：验证真实 config/default.yaml
# =====================================================================

class TestDefaultYAMLIntegration:
    """集成级测试 — 验证项目自带的 config/default.yaml 是否可正常加载和读取。"""

    @pytest.fixture
    def default_config(self):
        """加载项目根目录下的 config/default.yaml。"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(project_root, "config", "default.yaml")
        return ConfigManager(default_path)

    def test_loads_without_error(self, default_config):
        """真实配置文件应能正常加载。"""
        assert default_config.to_dict()

    def test_all_top_level_sections_exist(self, default_config):
        """所有模块分区 key 应存在。"""
        expected_sections = [
            "stream", "perception", "servo", "mavlink",
            "fsm", "logging", "camera", "gripper",
        ]
        data = default_config.to_dict()
        for section in expected_sections:
            assert section in data, f"缺少顶层配置段: {section}"

    def test_servo_kp_values(self, default_config):
        """伺服控制器 PD 增益应为设计文档指定的默认值（cargo_align 阶段）。"""
        assert default_config.get("servo.cargo_align.kp.x") == 0.003
        assert default_config.get("servo.cargo_align.kp.y") == 0.003
        assert default_config.get("servo.cargo_align.kp.yaw") == 0.6

    def test_fsm_tick_rate(self, default_config):
        """主循环频率应为 15 Hz。"""
        assert default_config.get("fsm.tick_rate_hz") == 15

    def test_stream_url(self, default_config):
        """双摄像头视频流地址应已配置。"""
        cargo_url = default_config.get("stream.cargo_cam.url")
        assert cargo_url is not None
        assert cargo_url.startswith("http://")
        delivery_url = default_config.get("stream.delivery_cam.url")
        assert delivery_url is not None
        assert delivery_url.startswith("http://")
