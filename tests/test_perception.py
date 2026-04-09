import math
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from core.perception import TargetPoseEstimator


# =====================================================================
#  辅助工具：构造模拟的 YOLO OBB 推理结果
# =====================================================================

def make_obb_result(detections: list[dict]):
    """
    构造 mock 的 ultralytics 推理结果。

    Args:
        detections: 检测结果列表，每个元素为:
            {"cx", "cy", "w", "h", "theta", "conf", "cls"}
            若为空列表，返回 obb=None 的结果。
    """
    result = MagicMock()

    if not detections:
        result.obb = None
        return [result]

    n = len(detections)
    xywhr_data = np.array(
        [[d["cx"], d["cy"], d["w"], d["h"], d["theta"]] for d in detections],
        dtype=np.float32,
    )
    conf_data = np.array([d["conf"] for d in detections], dtype=np.float32)
    cls_data = np.array([d["cls"] for d in detections], dtype=np.float32)

    obb = MagicMock()
    obb.xywhr = torch.tensor(xywhr_data)
    obb.conf = torch.tensor(conf_data)
    obb.cls = torch.tensor(cls_data)
    obb.__len__ = lambda self: n

    result.obb = obb
    return [result]


def make_empty_obb_result():
    """构造 obb 存在但检测数为 0 的结果。"""
    result = MagicMock()
    obb = MagicMock()
    obb.xywhr = torch.zeros((0, 5))
    obb.conf = torch.zeros((0,))
    obb.cls = torch.zeros((0,))
    obb.__len__ = lambda self: 0
    result.obb = obb
    return [result]


# =====================================================================
#  Fixtures
# =====================================================================

@pytest.fixture
def mock_weights(tmp_path):
    """创建一个临时权重文件路径。"""
    path = tmp_path / "fake_model.pt"
    path.touch()
    return str(path)


@pytest.fixture
def frame():
    """640×480 全黑帧。"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def estimator(mock_weights):
    """使用 mock YOLO 创建 TargetPoseEstimator 实例。"""
    with patch("core.perception.YOLO") as MockYOLO:
        mock_model = MagicMock()
        # 预热推理返回空结果
        mock_model.return_value = make_obb_result([])
        MockYOLO.return_value = mock_model

        est = TargetPoseEstimator(
            weights_path=mock_weights,
            conf_threshold=0.6,
            device="cpu",
        )
        # 重置 mock 调用计数（预热已完成）
        mock_model.reset_mock()
        return est


# =====================================================================
#  T1–T3：构造函数
# =====================================================================

class TestInit:
    """构造函数测试。"""

    def test_t1_valid_init(self, mock_weights):
        """T1 — 合法权重路径初始化成功，预热被调用。"""
        with patch("core.perception.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.return_value = make_obb_result([])
            MockYOLO.return_value = mock_model

            est = TargetPoseEstimator(mock_weights, conf_threshold=0.6, device="cpu")

            assert est is not None
            MockYOLO.assert_called_once_with(mock_weights)
            # 预热推理应被调用一次
            mock_model.assert_called_once()

    def test_t2_weights_not_found(self):
        """T2 — 权重文件不存在 → FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="权重文件不存在"):
            TargetPoseEstimator("/nonexistent/path/model.pt")

    def test_t3_custom_params(self, mock_weights):
        """T3 — 自定义 conf_threshold 和 device。"""
        with patch("core.perception.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.return_value = make_obb_result([])
            MockYOLO.return_value = mock_model

            est = TargetPoseEstimator(
                mock_weights, conf_threshold=0.8, device="cpu",
            )
            assert est._conf_threshold == 0.8
            assert est._device == "cpu"


# =====================================================================
#  T4–T7：process_frame — 正常检测
# =====================================================================

class TestProcessFrameNormal:
    """process_frame 正常检测测试。"""

    def test_t4_single_target_cls0(self, estimator, frame):
        """T4 — 单目标检测成功（target_cls_id=0）。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 320, "cy": 240, "w": 100, "h": 60, "theta": 0.1, "conf": 0.85, "cls": 0},
        ])

        result = estimator.process_frame(frame, target_cls_id=0)

        assert result is not None
        assert result["u"] == pytest.approx(320.0)
        assert result["v"] == pytest.approx(240.0)
        assert result["conf"] == pytest.approx(0.85, abs=1e-2)

    def test_t5_single_target_cls1(self, estimator, frame):
        """T5 — 单目标检测成功（target_cls_id=1）。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 400, "cy": 300, "w": 80, "h": 80, "theta": 0.3, "conf": 0.92, "cls": 1},
        ])

        result = estimator.process_frame(frame, target_cls_id=1)

        assert result is not None
        assert result["u"] == pytest.approx(400.0)
        assert result["v"] == pytest.approx(300.0)
        assert result["conf"] == pytest.approx(0.92, abs=1e-2)

    def test_t6_multiple_same_class_select_highest_conf(self, estimator, frame):
        """T6 — 多目标同类别 → 选择最高 conf。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 100, "cy": 100, "w": 50, "h": 50, "theta": 0.0, "conf": 0.70, "cls": 0},
            {"cx": 320, "cy": 240, "w": 90, "h": 60, "theta": 0.1, "conf": 0.95, "cls": 0},
            {"cx": 500, "cy": 400, "w": 40, "h": 40, "theta": 0.2, "conf": 0.80, "cls": 0},
        ])

        result = estimator.process_frame(frame, target_cls_id=0)

        assert result is not None
        assert result["u"] == pytest.approx(320.0)
        assert result["v"] == pytest.approx(240.0)
        assert result["conf"] == pytest.approx(0.95, abs=1e-2)

    def test_t7_return_dict_has_all_keys(self, estimator, frame):
        """T7 — 返回字典包含全部 6 个键。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 320, "cy": 240, "w": 100, "h": 60, "theta": 0.1, "conf": 0.85, "cls": 0},
        ])

        result = estimator.process_frame(frame, target_cls_id=0)

        assert result is not None
        expected_keys = {"u", "v", "theta", "conf", "w", "h"}
        assert set(result.keys()) == expected_keys


# =====================================================================
#  T8–T12：process_frame — 返回 None 的场景
# =====================================================================

class TestProcessFrameNone:
    """process_frame 返回 None 的场景测试。"""

    def test_t8_obb_is_none(self, estimator, frame):
        """T8 — YOLO 推理结果 obb 为 None。"""
        estimator._model.return_value = make_obb_result([])

        result = estimator.process_frame(frame, target_cls_id=0)
        assert result is None

    def test_t9_obb_empty(self, estimator, frame):
        """T9 — obb 存在但检测数为 0。"""
        estimator._model.return_value = make_empty_obb_result()

        result = estimator.process_frame(frame, target_cls_id=0)
        assert result is None

    def test_t10_class_mismatch(self, estimator, frame):
        """T10 — 检测到目标但类别不匹配。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 320, "cy": 240, "w": 100, "h": 60, "theta": 0.1, "conf": 0.9, "cls": 1},
        ])

        result = estimator.process_frame(frame, target_cls_id=0)
        assert result is None

    def test_t11_below_conf_threshold(self, estimator, frame):
        """T11 — 类别匹配但置信度低于阈值。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 320, "cy": 240, "w": 100, "h": 60, "theta": 0.1, "conf": 0.3, "cls": 0},
        ])

        result = estimator.process_frame(frame, target_cls_id=0)
        assert result is None

    def test_t12_mixed_classes_target_below_threshold(self, estimator, frame):
        """T12 — 多类别混合，目标类别全部低于阈值。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 320, "cy": 240, "w": 100, "h": 60, "theta": 0.1, "conf": 0.9, "cls": 1},
            {"cx": 100, "cy": 100, "w": 50, "h": 50, "theta": 0.0, "conf": 0.3, "cls": 0},
            {"cx": 200, "cy": 200, "w": 60, "h": 40, "theta": 0.2, "conf": 0.5, "cls": 0},
        ])

        result = estimator.process_frame(frame, target_cls_id=0)
        assert result is None


# =====================================================================
#  T13–T15：角度归一化验证
# =====================================================================

class TestAngleNormalization:
    """角度归一化验证（端到端，不 mock normalize_obb_angle）。"""

    def test_t13_theta_zero(self, estimator, frame):
        """T13 — theta=0 → 归一化后仍为 0。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 320, "cy": 240, "w": 100, "h": 60, "theta": 0.0, "conf": 0.9, "cls": 0},
        ])

        result = estimator.process_frame(frame, target_cls_id=0)
        assert result["theta"] == pytest.approx(0.0, abs=1e-10)

    def test_t14_theta_half_pi(self, estimator, frame):
        """T14 — theta=π/2 → C₂ 折叠后为 0。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 320, "cy": 240, "w": 100, "h": 60,
             "theta": math.pi / 2, "conf": 0.9, "cls": 0},
        ])

        result = estimator.process_frame(frame, target_cls_id=0)
        assert result["theta"] == pytest.approx(0.0, abs=1e-6)

    def test_t15_theta_in_range(self, estimator, frame):
        """T15 — theta=π/3 → 折叠到 [-π/4, π/4] 内。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 320, "cy": 240, "w": 100, "h": 60,
             "theta": math.pi / 3, "conf": 0.9, "cls": 0},
        ])

        result = estimator.process_frame(frame, target_cls_id=0)
        assert abs(result["theta"]) <= math.pi / 4 + 1e-10


# =====================================================================
#  T16–T17：多类别隔离性
# =====================================================================

class TestClassIsolation:
    """多类别隔离性测试。"""

    def test_t16_request_cls0_ignores_cls1(self, estimator, frame):
        """T16 — 同帧检测到 cls=0 和 cls=1，请求 cls=0 → 仅返回 cls=0。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 100, "cy": 100, "w": 50, "h": 50, "theta": 0.0, "conf": 0.95, "cls": 1},
            {"cx": 320, "cy": 240, "w": 90, "h": 60, "theta": 0.1, "conf": 0.80, "cls": 0},
        ])

        result = estimator.process_frame(frame, target_cls_id=0)

        assert result is not None
        assert result["u"] == pytest.approx(320.0)
        assert result["conf"] == pytest.approx(0.80, abs=1e-2)

    def test_t17_request_cls1_ignores_cls0(self, estimator, frame):
        """T17 — 同帧检测到 cls=0 和 cls=1，请求 cls=1 → 仅返回 cls=1。"""
        estimator._model.return_value = make_obb_result([
            {"cx": 100, "cy": 100, "w": 50, "h": 50, "theta": 0.0, "conf": 0.70, "cls": 0},
            {"cx": 400, "cy": 300, "w": 80, "h": 80, "theta": 0.2, "conf": 0.88, "cls": 1},
        ])

        result = estimator.process_frame(frame, target_cls_id=1)

        assert result is not None
        assert result["u"] == pytest.approx(400.0)
        assert result["conf"] == pytest.approx(0.88, abs=1e-2)


# =====================================================================
#  T18：集成测试（从 default.yaml 读取参数）
# =====================================================================

class TestIntegrationWithConfig:
    """集成级测试 — 验证与 default.yaml 配置参数的兼容性。"""

    def test_t18_config_instantiation(self):
        """T18 — 用 default.yaml 参数构造（mock YOLO）。"""
        from utils.config_manager import ConfigManager

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(project_root, "config", "default.yaml")
        config = ConfigManager(default_path)

        weights_path = config.get("perception.weights")
        conf_threshold = config.get("perception.conf_threshold")
        device = config.get("perception.device")

        # 创建临时权重文件以通过路径校验
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_weights = f.name

        try:
            with patch("core.perception.YOLO") as MockYOLO:
                mock_model = MagicMock()
                mock_model.return_value = make_obb_result([])
                MockYOLO.return_value = mock_model

                est = TargetPoseEstimator(
                    weights_path=tmp_weights,
                    conf_threshold=conf_threshold,
                    device="cpu",  # 测试中使用 CPU
                )

                assert est is not None
                assert est._conf_threshold == conf_threshold
        finally:
            os.unlink(tmp_weights)
