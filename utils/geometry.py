import math


def normalize_obb_angle(theta: float, symmetry_order: int = 2) -> float:
    """
    将 OBB 输出角度折叠到最小等效区间。

    Args:
        theta:          OBB 输出偏转角（弧度）
        symmetry_order: 对称阶数。
                        2 → C₂ 对称（矩形纸盒），折叠到 [-π/4, π/4]
                        1 → 不折叠，仅归一化到 [-π/2, π/2]

    Returns:
        归一化后的角度（弧度）

    Raises:
        ValueError: symmetry_order 不为 1 或 2
    """
    if symmetry_order not in (1, 2):
        raise ValueError(
            f"symmetry_order 必须为 1 或 2，收到: {symmetry_order}"
        )

    # 归一化到 [-π/2, π/2]
    period = math.pi
    half = period / 2
    theta = (theta + half) % period - half

    # 若 C₂ 对称，进一步折叠到 [-π/4, π/4]
    if symmetry_order == 2:
        period2 = math.pi / 2
        half2 = period2 / 2
        theta = (theta + half2) % period2 - half2

    return theta


def pixel_to_body_error(
    u: float,
    v: float,
    center_u: float,
    center_v: float,
) -> tuple[float, float]:
    """
    将像素坐标误差映射为机体系 (前向, 侧向) 误差。

    E_x = center_v - v    （纵向像素差 → 前向误差）
    E_y = u - center_u    （横向像素差 → 侧向误差）

    Args:
        u:        目标像素横坐标
        v:        目标像素纵坐标
        center_u: 图像光学中心横坐标（默认 320）
        center_v: 图像光学中心纵坐标（默认 240）

    Returns:
        (error_x, error_y) — 机体系前向和侧向误差（像素单位）
    """
    error_x = center_v - v
    error_y = u - center_u
    return (error_x, error_y)


def apply_deadband(value: float, threshold: float) -> float:
    """
    死区滤波器。

    Args:
        value:     输入值（误差或控制量）
        threshold: 死区阈值（≥ 0）

    Returns:
        滤波后的值

    Raises:
        ValueError: threshold < 0
    """
    if threshold < 0:
        raise ValueError(
            f"threshold 必须 ≥ 0，收到: {threshold}"
        )

    if abs(value) <= threshold:
        return 0.0
    return value


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    硬饱和限幅。

    Args:
        value:   输入值
        min_val: 下限
        max_val: 上限

    Returns:
        限幅后的值

    Raises:
        ValueError: min_val > max_val
    """
    if min_val > max_val:
        raise ValueError(
            f"min_val ({min_val}) 不得大于 max_val ({max_val})"
        )

    return max(min_val, min(value, max_val))
