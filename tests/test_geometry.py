import math
import os

import pytest

from utils.geometry import normalize_obb_angle, pixel_to_body_error, apply_deadband, clamp


PI = math.pi


# =====================================================================
#  T1–T14：normalize_obb_angle 测试
# =====================================================================

class TestNormalizeObbAngleC2:
    """C₂ 对称折叠（symmetry_order=2）测试。"""

    def test_t1_zero_angle(self):
        """T1 — 零角度不变。"""
        assert normalize_obb_angle(0.0, symmetry_order=2) == pytest.approx(0.0)

    def test_t2_small_positive_in_range(self):
        """T2 — 小正角度在 [-π/4, π/4] 内不动。"""
        assert normalize_obb_angle(0.3, symmetry_order=2) == pytest.approx(0.3)

    def test_t3_small_negative_in_range(self):
        """T3 — 小负角度在 [-π/4, π/4] 内不动。"""
        assert normalize_obb_angle(-0.3, symmetry_order=2) == pytest.approx(-0.3)

    def test_t4_half_pi_folds_to_zero(self):
        """T4 — π/2 折叠到 0（C₂ 对称核心场景）。"""
        assert normalize_obb_angle(PI / 2, symmetry_order=2) == pytest.approx(0.0, abs=1e-10)

    def test_t5_neg_half_pi_folds_to_zero(self):
        """T5 — -π/2 折叠到 0。"""
        assert normalize_obb_angle(-PI / 2, symmetry_order=2) == pytest.approx(0.0, abs=1e-10)

    def test_t6_pi_folds_to_zero(self):
        """T6 — π 折叠到 0（旋转 π 等效于 0）。"""
        assert normalize_obb_angle(PI, symmetry_order=2) == pytest.approx(0.0, abs=1e-10)

    def test_t7_quarter_pi_plus_epsilon(self):
        """T7 — π/4 + ε 超边界后翻折到 ≈ -π/4 + ε。"""
        eps = 0.01
        result = normalize_obb_angle(PI / 4 + eps, symmetry_order=2)
        expected = -PI / 4 + eps
        assert result == pytest.approx(expected, abs=1e-10)

    def test_t8_neg_quarter_pi_minus_epsilon(self):
        """T8 — -π/4 - ε 超边界后翻折到 ≈ π/4 - ε。"""
        eps = 0.01
        result = normalize_obb_angle(-PI / 4 - eps, symmetry_order=2)
        expected = PI / 4 - eps
        assert result == pytest.approx(expected, abs=1e-10)

    def test_t13_two_pi(self):
        """T13 — 2π 多周期折叠到 0。"""
        assert normalize_obb_angle(2 * PI, symmetry_order=2) == pytest.approx(0.0, abs=1e-10)

    def test_t14_neg_three_pi(self):
        """T14 — -3π 多周期折叠到 0。"""
        assert normalize_obb_angle(-3 * PI, symmetry_order=2) == pytest.approx(0.0, abs=1e-10)


class TestNormalizeObbAngleNoFold:
    """不折叠模式（symmetry_order=1）测试。"""

    def test_t9_no_fold_in_range(self):
        """T9 — order=1 时，π/4+0.01 在 [-π/2, π/2] 内不变。"""
        val = PI / 4 + 0.01
        result = normalize_obb_angle(val, symmetry_order=1)
        assert result == pytest.approx(val, abs=1e-10)

    def test_t10_pi_normalizes_to_zero(self):
        """T10 — order=1 时，π 归一化到 ≈ 0（或 -π/2 边界附近）。"""
        result = normalize_obb_angle(PI, symmetry_order=1)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_t11_neg_pi_normalizes_to_zero(self):
        """T11 — order=1 时，-π 归一化到 ≈ 0。"""
        result = normalize_obb_angle(-PI, symmetry_order=1)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestNormalizeObbAngleValidation:
    """参数校验测试。"""

    def test_t12_invalid_symmetry_order(self):
        """T12 — 非法 symmetry_order 抛出 ValueError。"""
        with pytest.raises(ValueError, match="symmetry_order 必须为 1 或 2"):
            normalize_obb_angle(0.0, symmetry_order=3)

    def test_invalid_symmetry_order_zero(self):
        """symmetry_order=0 也应抛出 ValueError。"""
        with pytest.raises(ValueError):
            normalize_obb_angle(0.0, symmetry_order=0)

    def test_invalid_symmetry_order_negative(self):
        """symmetry_order=-1 也应抛出 ValueError。"""
        with pytest.raises(ValueError):
            normalize_obb_angle(0.0, symmetry_order=-1)


# =====================================================================
#  T15–T22：pixel_to_body_error 测试
# =====================================================================

class TestPixelToBodyError:
    """像素误差到机体系误差映射测试。"""

    # 默认光学中心：640×480 → center=(320, 240)
    CU, CV = 320.0, 240.0

    def test_t15_target_at_center(self):
        """T15 — 目标在图像中心 → 零误差。"""
        ex, ey = pixel_to_body_error(self.CU, self.CV, self.CU, self.CV)
        assert ex == 0.0
        assert ey == 0.0

    def test_t16_target_above(self):
        """T16 — 目标偏上（v < center_v）→ 需前飞（E_x > 0）。"""
        ex, ey = pixel_to_body_error(320, 200, self.CU, self.CV)
        assert ex == 40.0
        assert ey == 0.0

    def test_t17_target_below(self):
        """T17 — 目标偏下（v > center_v）→ 需后退（E_x < 0）。"""
        ex, ey = pixel_to_body_error(320, 280, self.CU, self.CV)
        assert ex == -40.0
        assert ey == 0.0

    def test_t18_target_right(self):
        """T18 — 目标偏右（u > center_u）→ 需右移（E_y > 0）。"""
        ex, ey = pixel_to_body_error(360, 240, self.CU, self.CV)
        assert ex == 0.0
        assert ey == 40.0

    def test_t19_target_left(self):
        """T19 — 目标偏左（u < center_u）→ 需左移（E_y < 0）。"""
        ex, ey = pixel_to_body_error(280, 240, self.CU, self.CV)
        assert ex == 0.0
        assert ey == -40.0

    def test_t20_diagonal_offset(self):
        """T20 — 对角偏移（右上）→ 前飞 + 右移。"""
        ex, ey = pixel_to_body_error(360, 200, self.CU, self.CV)
        assert ex == 40.0
        assert ey == 40.0

    def test_t21_extreme_corner(self):
        """T21 — 极端角落 (0, 0)。"""
        ex, ey = pixel_to_body_error(0, 0, self.CU, self.CV)
        assert ex == 240.0
        assert ey == -320.0

    def test_t22_non_standard_center(self):
        """T22 — 非标准中心（320×240 分辨率，center=(160, 120)）。"""
        ex, ey = pixel_to_body_error(100, 100, 160, 120)
        assert ex == 20.0
        assert ey == -60.0


# =====================================================================
#  T23–T31：apply_deadband 测试
# =====================================================================

class TestApplyDeadband:
    """死区函数测试。"""

    def test_t23_value_in_deadband(self):
        """T23 — 值在死区内 → 输出 0。"""
        assert apply_deadband(10, 15) == 0.0

    def test_t24_value_exceeds_deadband(self):
        """T24 — 正值超出死区 → 输出原值。"""
        assert apply_deadband(20, 15) == 20.0

    def test_t25_negative_exceeds_deadband(self):
        """T25 — 负值超出死区 → 输出原值。"""
        assert apply_deadband(-20, 15) == -20.0

    def test_t26_value_equals_threshold(self):
        """T26 — 值等于阈值 → 边界归零（<= 语义）。"""
        assert apply_deadband(15, 15) == 0.0

    def test_t27_neg_value_equals_threshold(self):
        """T27 — 负值绝对值等于阈值 → 边界归零。"""
        assert apply_deadband(-15, 15) == 0.0

    def test_t28_zero_value(self):
        """T28 — 零值在死区内 → 输出 0。"""
        assert apply_deadband(0, 15) == 0.0

    def test_t29_zero_threshold_nonzero_value(self):
        """T29 — 零阈值，非零值 → 直接通过。"""
        assert apply_deadband(5, 0) == 5.0

    def test_t30_zero_threshold_zero_value(self):
        """T30 — 零阈值，零值 → 输出 0。"""
        assert apply_deadband(0, 0) == 0.0

    def test_t31_negative_threshold(self):
        """T31 — 负阈值 → 抛出 ValueError。"""
        with pytest.raises(ValueError, match="threshold 必须 ≥ 0"):
            apply_deadband(5, -1)


# =====================================================================
#  T32–T39：clamp 测试
# =====================================================================

class TestClamp:
    """饱和限幅函数测试。"""

    def test_t32_exceeds_upper(self):
        """T32 — 超上限 → 钳制到上限。"""
        assert clamp(0.5, -0.3, 0.3) == 0.3

    def test_t33_exceeds_lower(self):
        """T33 — 超下限 → 钳制到下限。"""
        assert clamp(-0.5, -0.3, 0.3) == -0.3

    def test_t34_in_range(self):
        """T34 — 在范围内 → 输出原值。"""
        assert clamp(0.2, -0.3, 0.3) == 0.2

    def test_t35_at_upper_limit(self):
        """T35 — 等于上限 → 上限本身。"""
        assert clamp(0.3, -0.3, 0.3) == 0.3

    def test_t36_at_lower_limit(self):
        """T36 — 等于下限 → 下限本身。"""
        assert clamp(-0.3, -0.3, 0.3) == -0.3

    def test_t37_zero_value(self):
        """T37 — 零值在范围中间。"""
        assert clamp(0.0, -0.3, 0.3) == 0.0

    def test_t38_min_equals_max(self):
        """T38 — min == max 退化情况。"""
        assert clamp(100, 0, 0) == 0

    def test_t39_min_greater_than_max(self):
        """T39 — min > max → 抛出 ValueError。"""
        with pytest.raises(ValueError, match="min_val.*不得大于.*max_val"):
            clamp(0, 1, -1)


# =====================================================================
#  T40–T42：集成级测试（与 config/default.yaml 联动）
# =====================================================================

class TestIntegrationWithConfig:
    """集成级测试 — 验证几何函数与 default.yaml 配置参数的兼容性。"""

    @pytest.fixture
    def config(self):
        """加载项目 default.yaml 中的关键参数。"""
        from utils.config_manager import ConfigManager

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(project_root, "config", "default.yaml")
        return ConfigManager(default_path)

    def test_t40_deadband_with_config_params(self, config):
        """T40 — 使用 default.yaml 中的死区参数调用 apply_deadband（cargo_align 阶段）。"""
        db_x = config.get("servo.cargo_align.deadband.x")
        db_y = config.get("servo.cargo_align.deadband.y")
        db_yaw = config.get("servo.cargo_align.deadband.yaw")

        # 死区参数应为非负数值
        assert db_x is not None and db_x >= 0
        assert db_y is not None and db_y >= 0
        assert db_yaw is not None and db_yaw >= 0

        # 在死区内应返回 0
        assert apply_deadband(db_x - 1, db_x) == 0.0
        assert apply_deadband(db_y - 1, db_y) == 0.0
        assert apply_deadband(db_yaw / 2, db_yaw) == 0.0

        # 超出死区应返回原值
        assert apply_deadband(db_x + 1, db_x) == db_x + 1
        assert apply_deadband(db_y + 1, db_y) == db_y + 1

    def test_t41_clamp_with_config_params(self, config):
        """T41 — 使用 default.yaml 中的限幅参数调用 clamp（cargo_align 阶段）。"""
        max_vx = config.get("servo.cargo_align.max_vel.x")
        max_vy = config.get("servo.cargo_align.max_vel.y")
        max_vyaw = config.get("servo.cargo_align.max_vel.yaw")

        # 限幅参数应为正数
        assert max_vx is not None and max_vx > 0
        assert max_vy is not None and max_vy > 0
        assert max_vyaw is not None and max_vyaw > 0

        # 超限值应被钳制
        assert clamp(max_vx * 2, -max_vx, max_vx) == max_vx
        assert clamp(-max_vy * 2, -max_vy, max_vy) == -max_vy
        assert clamp(max_vyaw * 3, -max_vyaw, max_vyaw) == max_vyaw

        # 范围内应不变
        assert clamp(0.0, -max_vx, max_vx) == 0.0

    def test_t42_full_pipeline_m2_to_m3(self, config):
        """T42 — 模拟 M2→M3 完整数据流：推理输出 → 角度归一化 → 像素映射 → 死区 → 限幅。"""
        # 模拟 M2 推理输出（使用 cargo_cam 中心摄像头）
        raw_u, raw_v = 350.0, 210.0
        raw_theta = 1.2  # 较大角度，应被折叠

        # 读取配置（使用 cargo_cam 和 cargo_align 阶段参数）
        center_u = config.get("camera.cargo_cam.center_u")
        center_v = config.get("camera.cargo_cam.center_v")
        db_x = config.get("servo.cargo_align.deadband.x")
        db_y = config.get("servo.cargo_align.deadband.y")
        db_yaw = config.get("servo.cargo_align.deadband.yaw")
        max_vx = config.get("servo.cargo_align.max_vel.x")
        max_vy = config.get("servo.cargo_align.max_vel.y")
        max_vyaw = config.get("servo.cargo_align.max_vel.yaw")
        kp_x = config.get("servo.cargo_align.kp.x")
        kp_y = config.get("servo.cargo_align.kp.y")
        kp_yaw = config.get("servo.cargo_align.kp.yaw")

        # Step 1：角度归一化（M2 输出处理）
        theta = normalize_obb_angle(raw_theta, symmetry_order=2)
        assert -PI / 4 <= theta <= PI / 4, f"归一化后角度 {theta} 不在 [-π/4, π/4] 内"

        # Step 2：像素→机体系误差映射
        error_x, error_y = pixel_to_body_error(raw_u, raw_v, center_u, center_v)
        assert error_x == center_v - raw_v  # 30.0
        assert error_y == raw_u - center_u  # 30.0

        # Step 3：死区滤波
        error_x_filtered = apply_deadband(error_x, db_x)
        error_y_filtered = apply_deadband(error_y, db_y)
        error_yaw_filtered = apply_deadband(theta, db_yaw)

        # error_x=30, db_x=15 → 超出死区，保留
        assert error_x_filtered == error_x
        assert error_y_filtered == error_y

        # Step 4：PD 控制（简化为 P 控制验证）
        vx_raw = kp_x * error_x_filtered
        vy_raw = kp_y * error_y_filtered
        vyaw_raw = kp_yaw * error_yaw_filtered

        # Step 5：饱和限幅
        vx = clamp(vx_raw, -max_vx, max_vx)
        vy = clamp(vy_raw, -max_vy, max_vy)
        vyaw = clamp(vyaw_raw, -max_vyaw, max_vyaw)

        # 验证输出在安全包线内
        assert -max_vx <= vx <= max_vx
        assert -max_vy <= vy <= max_vy
        assert -max_vyaw <= vyaw <= max_vyaw

        # 验证速度方向正确（目标偏右上 → vx > 0, vy > 0）
        assert vx > 0, "目标偏上，前向速度应为正"
        assert vy > 0, "目标偏右，侧向速度应为正"
