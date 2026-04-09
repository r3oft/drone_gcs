import math
import os
import tempfile

import cv2
import numpy as np
import pytest

from utils.visualization import DebugVisualizer, CLASS_COLORS


# =====================================================================
#  通用 Fixtures 与辅助工具
# =====================================================================

@pytest.fixture
def viz():
    """无录制、无快照的默认 DebugVisualizer。"""
    return DebugVisualizer()


@pytest.fixture
def black_frame():
    """640×480 全黑帧。"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def frame_modified(frame: np.ndarray) -> bool:
    """检查帧是否被修改（非全黑）。"""
    return np.any(frame != 0)


# =====================================================================
#  T1–T4：构造函数
# =====================================================================

class TestInit:
    """构造函数测试。"""

    def test_t1_default_init(self):
        """T1 — 默认参数初始化成功。"""
        viz = DebugVisualizer()
        assert viz is not None
        assert viz._recording is False

    def test_t2_with_record_path(self, tmp_path):
        """T2 — 指定录制路径 → _recording 为 True。"""
        path = str(tmp_path / "test.avi")
        viz = DebugVisualizer(record_path=path)
        assert viz._recording is True

    def test_t3_without_record_path(self):
        """T3 — 不指定录制路径 → _recording 为 False。"""
        viz = DebugVisualizer()
        assert viz._recording is False

    def test_t4_snapshot_dir_created(self, tmp_path):
        """T4 — 指定快照目录 → 目录被创建。"""
        snap_dir = str(tmp_path / "snapshots")
        assert not os.path.exists(snap_dir)

        viz = DebugVisualizer(snapshot_dir=snap_dir)
        assert os.path.isdir(snap_dir)
        assert viz._snapshot_dir == snap_dir


# =====================================================================
#  T5–T10：draw_obb 绘制验证
# =====================================================================

class TestDrawObb:
    """draw_obb 绘制验证。"""

    def test_t5_draws_on_black_frame(self, viz, black_frame):
        """T5 — 在黑色帧上绘制 OBB → 帧被修改。"""
        viz.draw_obb(black_frame, u=320, v=240, w=100, h=60, theta=0.0)
        assert frame_modified(black_frame)

    def test_t6_returns_same_reference(self, viz, black_frame):
        """T6 — 返回值是原帧引用（原地修改）。"""
        result = viz.draw_obb(black_frame, u=320, v=240, w=100, h=60, theta=0.0)
        assert result is black_frame

    def test_t7_different_angles(self, viz):
        """T7 — 不同角度的 OBB 绘制结果不同。"""
        frame_a = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_b = np.zeros((480, 640, 3), dtype=np.uint8)

        viz.draw_obb(frame_a, u=320, v=240, w=100, h=60, theta=0.0)
        viz.draw_obb(frame_b, u=320, v=240, w=100, h=60, theta=math.pi / 4)

        assert not np.array_equal(frame_a, frame_b)

    def test_t8_with_label_and_conf(self, viz, black_frame):
        """T8 — 带 label + conf 绘制 → 标签区域有文字。"""
        viz.draw_obb(
            black_frame, u=320, v=240, w=100, h=60, theta=0.0,
            label="cargo", conf=0.87,
        )
        assert frame_modified(black_frame)

    def test_t9_no_label_no_conf(self, viz):
        """T9 — 不带 label、conf=0 → 仅框和中心点。"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        viz.draw_obb(frame, u=320, v=240, w=100, h=60, theta=0.0, label="", conf=0.0)
        assert frame_modified(frame)

    def test_t10_custom_color(self, viz):
        """T10 — 自定义颜色（红色）→ 帧中包含红色像素。"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        red_bgr = (0, 0, 255)
        viz.draw_obb(frame, u=320, v=240, w=100, h=60, theta=0.0, color=red_bgr)

        # 检查帧中是否包含红色通道非零的像素
        assert np.any(frame[:, :, 2] > 0), "应包含红色像素"


# =====================================================================
#  T11–T13：draw_error_vector 绘制验证
# =====================================================================

class TestDrawErrorVector:
    """draw_error_vector 绘制验证。"""

    def test_t11_arrow_to_corner(self, viz, black_frame):
        """T11 — 中心到右下角的箭头 → 帧被修改。"""
        viz.draw_error_vector(black_frame, center=(320, 240), target=(500, 400))
        assert frame_modified(black_frame)

    def test_t12_zero_error(self, viz, black_frame):
        """T12 — center == target（零误差）→ 不崩溃。"""
        viz.draw_error_vector(black_frame, center=(320, 240), target=(320, 240))
        # 应绘制十字标记但不绘制箭头，不崩溃即可

    def test_t13_returns_same_reference(self, viz, black_frame):
        """T13 — 返回值是原帧引用。"""
        result = viz.draw_error_vector(black_frame, center=(320, 240), target=(400, 300))
        assert result is black_frame


# =====================================================================
#  T14–T17：draw_hud 绘制验证
# =====================================================================

class TestDrawHud:
    """draw_hud 绘制验证。"""

    def test_t14_full_info(self, viz, black_frame):
        """T14 — 完整 info 字典 → 帧左上角被修改。"""
        info = {
            "state": "TASK_REC_ALIGN", "camera": "cargo_cam",
            "target": "cargo", "vx": 0.12, "vy": -0.05,
            "vyaw": 0.08, "fps": 14.2, "dt": 0.067,
        }
        viz.draw_hud(black_frame, info)
        # 左上角 (0,0) 区域应被半透明背景修改
        assert frame_modified(black_frame)

    def test_t15_empty_info(self, viz, black_frame):
        """T15 — 空 info 字典 → 不崩溃，帧不变。"""
        original = black_frame.copy()
        viz.draw_hud(black_frame, {})
        assert np.array_equal(black_frame, original)

    def test_t16_partial_info(self, viz, black_frame):
        """T16 — 仅部分字段 → 不报错。"""
        viz.draw_hud(black_frame, {"state": "ALIGN", "fps": 15.0})
        assert frame_modified(black_frame)

    def test_t17_dual_class_info(self, viz, black_frame):
        """T17 — 包含双类别信息（camera + target）→ 帧被修改。"""
        viz.draw_hud(black_frame, {"camera": "delivery_cam", "target": "delivery_zone"})
        assert frame_modified(black_frame)


# =====================================================================
#  T18–T21：write_frame() 与 release()
# =====================================================================

class TestWriteFrameAndRelease:
    """write_frame() 与 release() 测试。"""

    def test_t18_write_multiple_frames(self, tmp_path):
        """T18 — write_frame() 多帧后 release() → 视频文件非空。"""
        video_path = str(tmp_path / "test.avi")
        viz = DebugVisualizer(record_path=video_path)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(10):
            cv2.circle(frame, (50 * i, 240), 20, (0, 255, 0), -1)
            viz.write_frame(frame)

        viz.release()

        assert os.path.isfile(video_path)
        assert os.path.getsize(video_path) > 0

    def test_t19_video_readable_after_release(self, tmp_path):
        """T19 — release() 后视频可被 VideoCapture 读取，帧数匹配。"""
        video_path = str(tmp_path / "test.avi")
        num_frames = 5
        viz = DebugVisualizer(record_path=video_path)

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        for _ in range(num_frames):
            viz.write_frame(frame)

        viz.release()

        cap = cv2.VideoCapture(video_path)
        count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            count += 1
        cap.release()

        assert count == num_frames

    def test_t20_release_idempotent(self, viz):
        """T20 — release() 后再次 release() → 幂等，不崩溃。"""
        viz.release()
        viz.release()
        viz.release()

    def test_t21_context_manager(self, tmp_path):
        """T21 — with 用法 → __exit__ 正确调用 release()。"""
        video_path = str(tmp_path / "test.avi")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with DebugVisualizer(record_path=video_path) as viz:
            viz.write_frame(frame)

        # with 块结束后，录制应已停止
        assert viz._recording is False
        assert os.path.isfile(video_path)


# =====================================================================
#  T22–T24：save_frame() 帧快照
# =====================================================================

class TestSaveFrame:
    """save_frame() 帧快照测试。"""

    def test_t22_save_creates_png(self, tmp_path):
        """T22 — 配置快照目录后保存帧 → PNG 文件被创建。"""
        snap_dir = str(tmp_path / "snaps")
        viz = DebugVisualizer(snapshot_dir=snap_dir)

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        viz.write_frame(frame)  # 递增 frame_count
        path = viz.save_frame(frame)

        assert path is not None
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_t23_save_with_tag(self, tmp_path):
        """T23 — 带 tag 保存 → 文件名包含 tag 字符串。"""
        snap_dir = str(tmp_path / "snaps")
        viz = DebugVisualizer(snapshot_dir=snap_dir)

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        viz.write_frame(frame)
        path = viz.save_frame(frame, tag="aligned")

        assert path is not None
        assert "aligned" in os.path.basename(path)
        assert path.endswith(".png")

    def test_t24_save_without_snapshot_dir(self, viz, black_frame):
        """T24 — 未配置快照目录时调用 → 返回 None，不报错。"""
        result = viz.save_frame(black_frame, tag="test")
        assert result is None


# =====================================================================
#  T25–T27：CLASS_COLORS 常量
# =====================================================================

class TestClassColors:
    """CLASS_COLORS 常量测试。"""

    def test_t25_pickup_zone_color(self):
        """T25 — pickup_zone 键存在且为浅绿色。"""
        assert "pickup_zone" in CLASS_COLORS
        assert CLASS_COLORS["pickup_zone"] == (50, 205, 50)

    def test_t26_delivery_zone_color(self):
        """T26 — delivery_zone 键存在且为蓝色。"""
        assert "delivery_zone" in CLASS_COLORS
        assert CLASS_COLORS["delivery_zone"] == (255, 180, 0)

    def test_t27_default_color(self):
        """T27 — default 回退色可用。"""
        assert "default" in CLASS_COLORS
        assert len(CLASS_COLORS["default"]) == 3
