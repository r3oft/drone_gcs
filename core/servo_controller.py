from utils.geometry import pixel_to_body_error, apply_deadband, clamp


class VisualServoController:
    """
    PD 视觉伺服控制器。

    将图像平面的像素/角度误差映射为机体坐标系下的期望速度向量。
    三轴（前向 x、侧向 y、偏航 yaw）完全解耦，各自独立 PD 控制。
    """

    _AXIS_COUNT = 3  # x, y, yaw
    _VALID_AXIS_MAPS = {"standard", "swap_xy"}

    def __init__(
        self,
        kp: list[float],
        kd: list[float],
        deadband: list[float],
        max_vel: list[float],
        axis_map: str = "standard",
        axis_sign: list[float] = None,
    ) -> None:
        """
        Args:
            kp:       比例增益 [kp_x, kp_y, kp_yaw]
            kd:       微分增益 [kd_x, kd_y, kd_yaw]
            deadband: 死区阈值 [db_x, db_y, db_yaw]（像素/弧度）
            max_vel:  饱和限幅 [max_vx, max_vy, max_vyaw]（m/s 或 rad/s）
            axis_map: 图像误差到机体系速度轴的映射：
                      "standard"：图像纵向 -> vx，图像横向 -> vy
                      "swap_xy"： 图像横向 -> vx，图像纵向 -> vy
            axis_sign: 各控制轴符号 [sign_x, sign_y, sign_yaw]，每项为 +1 或 -1

        Raises:
            ValueError: 列表长度不为 3 / deadband 含负值 / max_vel 含非正值
        """
        for name, param in [("kp", kp), ("kd", kd), ("deadband", deadband), ("max_vel", max_vel)]:
            if len(param) != self._AXIS_COUNT:
                raise ValueError(
                    f"{name} 长度必须为 {self._AXIS_COUNT}，收到长度 {len(param)}"
                )

        for i, val in enumerate(deadband):
            if val < 0:
                raise ValueError(
                    f"deadband[{i}] 必须 ≥ 0，收到: {val}"
                )

        for i, val in enumerate(max_vel):
            if val <= 0:
                raise ValueError(
                    f"max_vel[{i}] 必须 > 0，收到: {val}"
                )

        if axis_map not in self._VALID_AXIS_MAPS:
            raise ValueError(
                f"axis_map 必须是 {sorted(self._VALID_AXIS_MAPS)} 之一，收到: {axis_map!r}"
            )
        if axis_sign is None:
            axis_sign = [1.0, 1.0, 1.0]
        if len(axis_sign) != self._AXIS_COUNT:
            raise ValueError(
                f"axis_sign 长度必须为 {self._AXIS_COUNT}，收到长度 {len(axis_sign)}"
            )
        for i, val in enumerate(axis_sign):
            if float(val) not in (-1.0, 1.0):
                raise ValueError(
                    f"axis_sign[{i}] 必须为 +1 或 -1，收到: {val}"
                )

        self._kp: list[float] = list(kp)
        self._kd: list[float] = list(kd)
        self._deadband: list[float] = list(deadband)
        self._max_vel: list[float] = list(max_vel)
        self._axis_map = axis_map
        self._axis_sign: list[float] = [float(v) for v in axis_sign]

        self._prev_errors: list[float] = [0.0] * self._AXIS_COUNT

    def compute_velocity(
        self,
        target_pose: dict,
        center_u: float,
        center_v: float,
        dt: float,
    ) -> tuple[float, float, float]:
        """
        Args:
            target_pose: M2 输出字典 {"u", "v", "theta", "conf"}
            center_u:    图像光学中心横坐标
            center_v:    图像光学中心纵坐标
            dt:          距上一帧的时间间隔（秒），≤0 时跳过 D 项

        Returns:
            (vx, vy, omega_z) — 限幅后的机体系期望速度
        """
        debug = self.compute_debug(target_pose, center_u, center_v, dt)
        return debug["velocities"]

    def compute_debug(
        self,
        target_pose: dict,
        center_u: float,
        center_v: float,
        dt: float,
    ) -> dict:
        """Compute PD output and return intermediate terms for HUD/debugging."""
        error_forward, error_right = pixel_to_body_error(
            target_pose["u"], target_pose["v"],
            center_u, center_v,
        )
        error_yaw = -target_pose["theta"]

        if self._axis_map == "swap_xy":
            error_x, error_y = error_right, error_forward
        else:
            error_x, error_y = error_forward, error_right

        error_x *= self._axis_sign[0]
        error_y *= self._axis_sign[1]
        error_yaw *= self._axis_sign[2]

        raw_errors = [error_x, error_y, error_yaw]

        errors = [
            apply_deadband(raw_errors[i], self._deadband[i])
            for i in range(self._AXIS_COUNT)
        ]

        velocities = [0.0] * self._AXIS_COUNT
        p_terms = [0.0] * self._AXIS_COUNT
        d_terms = [0.0] * self._AXIS_COUNT
        for i in range(self._AXIS_COUNT):
            p_term = self._kp[i] * errors[i]

            if dt > 0:
                d_term = self._kd[i] * (errors[i] - self._prev_errors[i]) / dt
            else:
                d_term = 0.0

            p_terms[i] = p_term
            d_terms[i] = d_term
            velocities[i] = p_term + d_term

        self._prev_errors = list(errors)

        vx = clamp(velocities[0], -self._max_vel[0], self._max_vel[0])
        vy = clamp(velocities[1], -self._max_vel[1], self._max_vel[1])
        omega_z = clamp(velocities[2], -self._max_vel[2], self._max_vel[2])

        return {
            "raw_errors": tuple(raw_errors),
            "errors": tuple(errors),
            "p_terms": tuple(p_terms),
            "d_terms": tuple(d_terms),
            "velocities_raw": tuple(velocities),
            "velocities": (vx, vy, omega_z),
            "dt": dt,
        }

    def reset(self) -> None:
        self._prev_errors = [0.0] * self._AXIS_COUNT
