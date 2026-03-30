import math
import os

import pytest

from core.servo_controller import VisualServoController


# =====================================================================
#  通用 Fixtures 与辅助工具
# =====================================================================

# 默认参数（与 default.yaml 一致）
DEFAULT_KP = [0.003, 0.003, 0.6]
DEFAULT_KD = [0.001, 0.001, 0.1]
DEFAULT_DB = [15.0, 15.0, 0.05]
DEFAULT_MV = [0.3, 0.3, 0.5]

# 图像中心（640×480）
CU, CV = 320.0, 240.0


@pytest.fixture
def ctrl():
    """使用默认参数创建控制器实例（每个测试独立使用）。"""
    return VisualServoController(
        kp=DEFAULT_KP, kd=DEFAULT_KD,
        deadband=DEFAULT_DB, max_vel=DEFAULT_MV,
    )


@pytest.fixture
def ctrl_pure_p():
    """纯 P 控制器（kd 全零），用于精确数值验证。"""
    return VisualServoController(
        kp=DEFAULT_KP, kd=[0.0, 0.0, 0.0],
        deadband=[0.0, 0.0, 0.0], max_vel=DEFAULT_MV,
    )


def make_pose(u=CU, v=CV, theta=0.0, conf=0.9):
    """快速构造 target_pose 字典。"""
    return {"u": u, "v": v, "theta": theta, "conf": conf}


# =====================================================================
#  T1–T8：构造函数与参数校验
# =====================================================================

class TestInit:
    """构造函数参数校验测试。"""

    def test_t1_valid_params(self):
        """T1 — 合法参数初始化成功。"""
        ctrl = VisualServoController(
            kp=DEFAULT_KP, kd=DEFAULT_KD,
            deadband=DEFAULT_DB, max_vel=DEFAULT_MV,
        )
        assert ctrl is not None

    def test_t2_kp_wrong_length(self):
        """T2 — kp 长度不为 3。"""
        with pytest.raises(ValueError, match="kp"):
            VisualServoController(
                kp=[0.003, 0.003], kd=DEFAULT_KD,
                deadband=DEFAULT_DB, max_vel=DEFAULT_MV,
            )

    def test_t3_kd_wrong_length(self):
        """T3 — kd 长度不为 3。"""
        with pytest.raises(ValueError, match="kd"):
            VisualServoController(
                kp=DEFAULT_KP, kd=[0.001],
                deadband=DEFAULT_DB, max_vel=DEFAULT_MV,
            )

    def test_t4_deadband_wrong_length(self):
        """T4 — deadband 长度不为 3。"""
        with pytest.raises(ValueError, match="deadband"):
            VisualServoController(
                kp=DEFAULT_KP, kd=DEFAULT_KD,
                deadband=[15.0, 15.0, 0.05, 0.1], max_vel=DEFAULT_MV,
            )

    def test_t5_max_vel_wrong_length(self):
        """T5 — max_vel 长度不为 3。"""
        with pytest.raises(ValueError, match="max_vel"):
            VisualServoController(
                kp=DEFAULT_KP, kd=DEFAULT_KD,
                deadband=DEFAULT_DB, max_vel=[0.3, 0.3],
            )

    def test_t6_deadband_negative(self):
        """T6 — deadband 含负值。"""
        with pytest.raises(ValueError, match="deadband"):
            VisualServoController(
                kp=DEFAULT_KP, kd=DEFAULT_KD,
                deadband=[15.0, -1.0, 0.05], max_vel=DEFAULT_MV,
            )

    def test_t7_max_vel_zero(self):
        """T7 — max_vel 含零值。"""
        with pytest.raises(ValueError, match="max_vel"):
            VisualServoController(
                kp=DEFAULT_KP, kd=DEFAULT_KD,
                deadband=DEFAULT_DB, max_vel=[0.3, 0.0, 0.5],
            )

    def test_t8_max_vel_negative(self):
        """T8 — max_vel 含负值。"""
        with pytest.raises(ValueError, match="max_vel"):
            VisualServoController(
                kp=DEFAULT_KP, kd=DEFAULT_KD,
                deadband=DEFAULT_DB, max_vel=[0.3, -0.1, 0.5],
            )


# =====================================================================
#  T9–T13：compute_velocity 基础行为
# =====================================================================

class TestComputeVelocityBasic:
    """compute_velocity 基础方向性与零误差测试。"""

    def test_t9_target_at_center(self, ctrl):
        """T9 — 目标在图像中心，theta=0 → 输出 (0, 0, 0)。"""
        vx, vy, vz = ctrl.compute_velocity(make_pose(), CU, CV, dt=0.067)
        assert vx == 0.0
        assert vy == 0.0
        assert vz == 0.0

    def test_t10_target_above(self, ctrl):
        """T10 — 目标偏上（v < center_v）→ vx > 0（前飞）。"""
        # v=200, E_x = 240-200 = 40 > deadband(15)
        vx, vy, vz = ctrl.compute_velocity(make_pose(v=200), CU, CV, dt=0.067)
        assert vx > 0, "目标偏上，前向速度应为正"
        assert vy == 0.0, "横向无偏移"
        assert vz == 0.0, "偏航无偏移"

    def test_t11_target_right(self, ctrl):
        """T11 — 目标偏右（u > center_u）→ vy > 0（右移）。"""
        # u=360, E_y = 360-320 = 40 > deadband(15)
        vx, vy, vz = ctrl.compute_velocity(make_pose(u=360), CU, CV, dt=0.067)
        assert vx == 0.0, "纵向无偏移"
        assert vy > 0, "目标偏右，侧向速度应为正"
        assert vz == 0.0, "偏航无偏移"

    def test_t12_target_lower_left_with_yaw(self, ctrl):
        """T12 — 目标偏左下 + 正偏航 → vx < 0, vy < 0, omega_z < 0。"""
        # u=280 → E_y = 280-320 = -40; v=280 → E_x = 240-280 = -40; theta=0.3 → E_yaw = -0.3
        vx, vy, vz = ctrl.compute_velocity(make_pose(u=280, v=280, theta=0.3), CU, CV, dt=0.067)
        assert vx < 0, "目标偏下，前向速度应为负"
        assert vy < 0, "目标偏左，侧向速度应为负"
        assert vz < 0, "正偏航角，角速度修正应为负"

    def test_t13_yaw_only(self, ctrl):
        """T13 — 仅偏航误差（目标在中心，theta=0.3）→ vx=0, vy=0, omega_z<0。"""
        vx, vy, vz = ctrl.compute_velocity(make_pose(theta=0.3), CU, CV, dt=0.067)
        assert vx == 0.0
        assert vy == 0.0
        assert vz < 0, "正偏航角 → 负角速度修正"


# =====================================================================
#  T14–T17：死区行为
# =====================================================================

class TestDeadband:
    """死区滤波行为测试。"""

    def test_t14_pixel_error_in_deadband(self, ctrl):
        """T14 — 像素误差在死区内 → vx = 0。"""
        # v=230, E_x = 240-230 = 10 < deadband(15)
        vx, vy, vz = ctrl.compute_velocity(make_pose(v=230), CU, CV, dt=0.067)
        assert vx == 0.0, "E_x=10 在死区(15)内，应输出零"

    def test_t15_pixel_error_exceeds_deadband(self, ctrl):
        """T15 — 像素误差超出死区 → vx ≠ 0。"""
        # v=210, E_x = 240-210 = 30 > deadband(15)
        vx, vy, vz = ctrl.compute_velocity(make_pose(v=210), CU, CV, dt=0.067)
        assert vx != 0.0, "E_x=30 超出死区(15)，应有输出"

    def test_t16_yaw_error_in_deadband(self, ctrl):
        """T16 — 偏航误差在死区内（|theta|=0.03 < db=0.05）→ omega_z = 0。"""
        vx, vy, vz = ctrl.compute_velocity(make_pose(theta=0.03), CU, CV, dt=0.067)
        assert vz == 0.0, "|E_yaw|=0.03 在死区(0.05)内"

    def test_t17_all_errors_in_deadband(self, ctrl):
        """T17 — 全部误差在死区内 → 输出 (0, 0, 0)。"""
        # E_x=5, E_y=-5, E_yaw=0.02 — 全部小于各死区阈值
        vx, vy, vz = ctrl.compute_velocity(
            make_pose(u=315, v=235, theta=0.02), CU, CV, dt=0.067,
        )
        assert (vx, vy, vz) == (0.0, 0.0, 0.0), "所有误差在死区内，应输出全零"


# =====================================================================
#  T18–T20：饱和限幅行为
# =====================================================================

class TestSaturation:
    """饱和限幅行为测试。"""

    def test_t18_large_pixel_error_clamped(self, ctrl):
        """T18 — 大像素误差 → |vx| ≤ max_vx。"""
        # v=40, E_x = 240-40 = 200（非常大）
        vx, vy, vz = ctrl.compute_velocity(make_pose(v=40), CU, CV, dt=0.067)
        assert abs(vx) <= DEFAULT_MV[0] + 1e-10, f"|vx|={abs(vx)} 应 ≤ {DEFAULT_MV[0]}"

    def test_t19_large_yaw_error_clamped(self, ctrl):
        """T19 — 大偏航误差 → |omega_z| ≤ max_vyaw。"""
        # theta=0.7, E_yaw = -0.7
        vx, vy, vz = ctrl.compute_velocity(make_pose(theta=0.7), CU, CV, dt=0.067)
        assert abs(vz) <= DEFAULT_MV[2] + 1e-10, f"|omega_z|={abs(vz)} 应 ≤ {DEFAULT_MV[2]}"

    def test_t20_all_axes_clamped(self, ctrl):
        """T20 — 三轴全大误差 → 全部被限幅。"""
        vx, vy, vz = ctrl.compute_velocity(
            make_pose(u=640, v=0, theta=0.7), CU, CV, dt=0.067,
        )
        assert abs(vx) <= DEFAULT_MV[0] + 1e-10
        assert abs(vy) <= DEFAULT_MV[1] + 1e-10
        assert abs(vz) <= DEFAULT_MV[2] + 1e-10


# =====================================================================
#  T21–T25：微分项行为
# =====================================================================

class TestDerivativeTerm:
    """微分项（D 项）行为测试。"""

    def test_t21_first_frame_d_term(self):
        """T21 — 首帧调用：D 项基于 0→E 的变化率。"""
        ctrl_p = VisualServoController(
            kp=[0.003, 0.003, 0.6], kd=[0.001, 0.001, 0.1],
            deadband=[0.0, 0.0, 0.0], max_vel=[10.0, 10.0, 10.0],  # 大限幅避免截断
        )
        dt = 0.1
        error_x = 100.0  # v = 240 - 100 = 140 → E_x = 100

        vx, _, _ = ctrl_p.compute_velocity(make_pose(v=140), CU, CV, dt=dt)

        # 首帧：P = 0.003*100 = 0.3, D = 0.001*(100-0)/0.1 = 1.0
        expected = 0.003 * error_x + 0.001 * (error_x - 0.0) / dt
        assert vx == pytest.approx(expected, abs=1e-10)

    def test_t22_error_decreasing_d_negative(self):
        """T22 — 连续两帧，误差从大到小 → D 项为负（减速）。"""
        ctrl_nodz = VisualServoController(
            kp=[0.003, 0.003, 0.6], kd=[0.001, 0.001, 0.1],
            deadband=[0.0, 0.0, 0.0], max_vel=[10.0, 10.0, 10.0],
        )
        dt = 0.1

        # 第一帧：大误差
        vx1, _, _ = ctrl_nodz.compute_velocity(make_pose(v=140), CU, CV, dt=dt)  # E_x=100

        # 第二帧：小误差
        vx2, _, _ = ctrl_nodz.compute_velocity(make_pose(v=200), CU, CV, dt=dt)  # E_x=40

        # 第二帧的 D 项 = kd * (40 - 100) / 0.1 = 0.001 * (-600) = -0.6
        # P 项 = 0.003 * 40 = 0.12
        # 总计 = 0.12 - 0.6 = -0.48（D 项为负，产生减速效果）
        expected_vx2 = 0.003 * 40 + 0.001 * (40 - 100) / dt
        assert vx2 == pytest.approx(expected_vx2, abs=1e-10)
        # 验证 D 项确实为负
        d_term = 0.001 * (40 - 100) / dt
        assert d_term < 0

    def test_t23_error_unchanged_d_zero(self):
        """T23 — 连续两帧误差不变 → D 项为零。"""
        ctrl_nodz = VisualServoController(
            kp=[0.003, 0.003, 0.6], kd=[0.001, 0.001, 0.1],
            deadband=[0.0, 0.0, 0.0], max_vel=[10.0, 10.0, 10.0],
        )
        dt = 0.1

        # 两帧相同误差
        vx1, _, _ = ctrl_nodz.compute_velocity(make_pose(v=190), CU, CV, dt=dt)  # E_x=50
        vx2, _, _ = ctrl_nodz.compute_velocity(make_pose(v=190), CU, CV, dt=dt)  # E_x=50

        # 第二帧 D 项 = kd * (50-50)/0.1 = 0
        # P 项 = 0.003 * 50 = 0.15
        expected = 0.003 * 50  # 纯 P
        assert vx2 == pytest.approx(expected, abs=1e-10)

    def test_t24_dt_zero_skips_d_term(self):
        """T24 — dt=0 防护：D 项归零，仅输出 P 项。"""
        ctrl_nodz = VisualServoController(
            kp=[0.003, 0.003, 0.6], kd=[0.001, 0.001, 0.1],
            deadband=[0.0, 0.0, 0.0], max_vel=[10.0, 10.0, 10.0],
        )

        vx, _, _ = ctrl_nodz.compute_velocity(make_pose(v=140), CU, CV, dt=0.0)

        # dt=0 → D 项跳过，仅 P 项
        expected = 0.003 * 100  # E_x = 100
        assert vx == pytest.approx(expected, abs=1e-10)

    def test_t25_dt_negative_skips_d_term(self):
        """T25 — dt 为负 → D 项归零，仅输出 P 项。"""
        ctrl_nodz = VisualServoController(
            kp=[0.003, 0.003, 0.6], kd=[0.001, 0.001, 0.1],
            deadband=[0.0, 0.0, 0.0], max_vel=[10.0, 10.0, 10.0],
        )

        vx, _, _ = ctrl_nodz.compute_velocity(make_pose(v=140), CU, CV, dt=-0.05)

        expected = 0.003 * 100
        assert vx == pytest.approx(expected, abs=1e-10)


# =====================================================================
#  T26–T27：reset() 行为
# =====================================================================

class TestReset:
    """reset() 状态清零行为测试。"""

    def test_t26_reset_clears_prev_errors(self):
        """T26 — reset 后等效于全新首帧。"""
        ctrl_nodz = VisualServoController(
            kp=[0.003, 0.003, 0.6], kd=[0.001, 0.001, 0.1],
            deadband=[0.0, 0.0, 0.0], max_vel=[10.0, 10.0, 10.0],
        )
        dt = 0.1

        # 第一帧
        vx_first, _, _ = ctrl_nodz.compute_velocity(make_pose(v=190), CU, CV, dt=dt)

        # 干扰几帧
        ctrl_nodz.compute_velocity(make_pose(v=140), CU, CV, dt=dt)
        ctrl_nodz.compute_velocity(make_pose(v=100), CU, CV, dt=dt)

        # reset
        ctrl_nodz.reset()

        # reset 后重新以相同输入调用，应与第一帧结果一致
        vx_after_reset, _, _ = ctrl_nodz.compute_velocity(make_pose(v=190), CU, CV, dt=dt)

        assert vx_after_reset == pytest.approx(vx_first, abs=1e-10)

    def test_t27_reset_idempotent(self, ctrl):
        """T27 — 连续 reset 不报错（幂等操作）。"""
        ctrl.reset()
        ctrl.reset()
        ctrl.reset()
        # 不应抛出异常


# =====================================================================
#  T28–T29：纯 P 控制器精确数值验证（kd=0）
# =====================================================================

class TestPureProportional:
    """纯 P 控制器（kd=0）精确数值验证。"""

    def test_t28_single_axis_pure_p(self, ctrl_pure_p):
        """T28 — 单轴纯 P 验证：vx = kp_x * E_x（精确数值）。"""
        # E_x = 240 - 190 = 50, kp_x = 0.003 → vx = 0.15
        vx, vy, vz = ctrl_pure_p.compute_velocity(make_pose(v=190), CU, CV, dt=0.067)
        assert vx == pytest.approx(0.003 * 50, abs=1e-10)

    def test_t29_three_axis_pure_p(self, ctrl_pure_p):
        """T29 — 三轴同时纯 P 验证：无交叉耦合。"""
        # E_x = 240-190 = 50, E_y = 370-320 = 50, E_yaw = -0.2
        vx, vy, vz = ctrl_pure_p.compute_velocity(
            make_pose(u=370, v=190, theta=0.2), CU, CV, dt=0.067,
        )
        assert vx == pytest.approx(0.003 * 50, abs=1e-10)
        assert vy == pytest.approx(0.003 * 50, abs=1e-10)
        assert vz == pytest.approx(0.6 * (-0.2), abs=1e-10)


# =====================================================================
#  T30–T32：与 config/default.yaml 集成测试
# =====================================================================

class TestIntegrationWithConfig:
    """集成级测试 — 验证与 default.yaml 配置参数的兼容性。"""

    @pytest.fixture
    def config_ctrl(self):
        """使用 default.yaml 参数构造控制器。"""
        from utils.config_manager import ConfigManager

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(project_root, "config", "default.yaml")
        config = ConfigManager(default_path)

        return VisualServoController(
            kp=[config.get("servo.cargo_align.kp.x"), config.get("servo.cargo_align.kp.y"), config.get("servo.cargo_align.kp.yaw")],
            kd=[config.get("servo.cargo_align.kd.x"), config.get("servo.cargo_align.kd.y"), config.get("servo.cargo_align.kd.yaw")],
            deadband=[config.get("servo.cargo_align.deadband.x"), config.get("servo.cargo_align.deadband.y"), config.get("servo.cargo_align.deadband.yaw")],
            max_vel=[config.get("servo.cargo_align.max_vel.x"), config.get("servo.cargo_align.max_vel.y"), config.get("servo.cargo_align.max_vel.yaw")],
        )

    def test_t30_config_instantiation(self, config_ctrl):
        """T30 — 用 default.yaml 参数构造控制器成功。"""
        assert config_ctrl is not None

    def test_t31_full_pipeline_m2_to_m3(self, config_ctrl):
        """T31 — 模拟 M2→M3 完整 Pipeline：输出方向正确、幅值在安全包线内。"""
        # 模拟 M2 输出：目标偏右上，有轻微偏航
        target = {"u": 380.0, "v": 180.0, "theta": 0.15, "conf": 0.85}
        vx, vy, vz = config_ctrl.compute_velocity(target, 320.0, 240.0, dt=0.067)

        # 方向验证
        assert vx > 0, "目标偏上 → 前飞"
        assert vy > 0, "目标偏右 → 右移"
        assert vz < 0, "正偏航 → 逆时针修正"

        # 安全包线验证
        assert abs(vx) <= 0.3
        assert abs(vy) <= 0.3
        assert abs(vz) <= 0.5

    def test_t32_alignment_complete_all_zero(self, config_ctrl):
        """T32 — 对齐完成场景：误差全在死区内 → 输出全零 → M5 可判定防抖跃迁完成。"""
        # 误差极小：E_x=3, E_y=-2, E_yaw=0.01 — 全部小于各死区
        target = {"u": 318.0, "v": 237.0, "theta": 0.01, "conf": 0.92}
        vx, vy, vz = config_ctrl.compute_velocity(target, 320.0, 240.0, dt=0.067)

        assert vx == 0.0 and vy == 0.0 and vz == 0.0, \
            "误差全在死区内，输出应为全零（M5 防抖跃迁判定依据）"
