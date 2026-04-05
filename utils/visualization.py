import math
import os

import cv2
import numpy as np


CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "cargo":        (0, 255, 0),     # 绿色
    "landing_zone": (255, 180, 0),   # 亮蓝色
    "default":      (0, 200, 255),   # 橙色
}


class DebugVisualizer:
    """
    在图像帧上叠绘 OBB 旋转框、误差向量箭头、FSM 状态等信息，
    用于训练数据验证、离线仿真和实飞调试。

    输出机制：
        - 视频录制：write_frame() → .avi 文件，事后回看
        - 帧快照：save_frame() → .png 文件，关键时刻高清存档
    """

    _HUD_LINE_HEIGHT = 22
    _HUD_PAD = 8
    _HUD_WIDTH = 260
    _HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
    _HUD_FONT_SCALE = 0.5
    _HUD_FONT_THICKNESS = 1
    _RECORD_FPS = 15

    def __init__(
        self,
        record_path: str | None = None,
        snapshot_dir: str | None = None,
    ) -> None:
        """
        Args:
            record_path:  视频录制输出路径（.avi 格式）。
                          为 None 时不录制。
            snapshot_dir: 帧快照保存目录。
                          为 None 时不支持 save_frame()。
        """
        # ── 视频录制 ──
        self._record_path = record_path
        self._writer: cv2.VideoWriter | None = None  # 延迟初始化
        self._recording = record_path is not None

        # ── 帧快照 ──
        self._snapshot_dir = snapshot_dir
        if snapshot_dir is not None:
            os.makedirs(snapshot_dir, exist_ok=True)

        # ── 帧计数器 ──
        self._frame_count = 0


    def draw_obb(
        self,
        frame: np.ndarray,
        u: float, v: float,
        w: float, h: float,
        theta: float,
        label: str = "",
        conf: float = 0.0,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        在画面上绘制旋转边界框 + 中心点 + 类别/置信度标签。

        Args:
            frame: 输入图像（BGR，原地修改）
            u, v:  OBB 中心像素坐标
            w, h:  OBB 宽高（像素）
            theta: OBB 旋转角（弧度）
            label: 类别名称（为空时不显示文字）
            conf:  检测置信度（0 时不显示）
            color: BGR 颜色

        Returns:
            修改后的 frame（同一引用）
        """
        # 计算旋转矩形的四个顶点
        angle_deg = math.degrees(theta)
        rect = ((u, v), (w, h), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # 绘制四边形轮廓
        cv2.drawContours(frame, [box], 0, color, 2)

        # 绘制中心点
        cv2.circle(frame, (int(u), int(v)), 4, color, -1)

        # 绘制类别 + 置信度标签
        if label or conf > 0:
            text_parts = []
            if label:
                text_parts.append(label)
            if conf > 0:
                text_parts.append(f"{conf:.0%}")
            text = " ".join(text_parts)

            text_x = int(u - w / 2)
            text_y = int(v - h / 2 - 8)

            (tw, th_text), _ = cv2.getTextSize(
                text, self._HUD_FONT, self._HUD_FONT_SCALE, self._HUD_FONT_THICKNESS,
            )
            # 文字背景矩形
            cv2.rectangle(
                frame,
                (text_x, text_y - th_text - 4),
                (text_x + tw + 4, text_y),
                color, -1,
            )
            # 文字（黑色，在彩色背景上）
            cv2.putText(
                frame, text, (text_x + 2, text_y - 2),
                self._HUD_FONT, self._HUD_FONT_SCALE,
                (0, 0, 0), self._HUD_FONT_THICKNESS, cv2.LINE_AA,
            )

        return frame


    def draw_error_vector(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        target: tuple[int, int],
        color: tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        """
        Args:
            frame:  输入图像（BGR，原地修改）
            center: 光学中心坐标 (cu, cv)
            target: 目标中心坐标 (u, v)
            color:  BGR 颜色

        Returns:
            修改后的 frame
        """
        cx, cy = center

        cross_size = 8
        cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), color, 1)
        cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), color, 1)
        if center != target:
            cv2.arrowedLine(frame, center, target, color, 2, tipLength=0.05)

        return frame


    def draw_hud(
        self,
        frame: np.ndarray,
        info: dict,
    ) -> np.ndarray:
        """
        Args:
            frame: 输入图像（BGR，原地修改）
            info:  HUD 信息字典

        Returns:
            修改后的 frame
        """
        lines = []
        if "state" in info:
            lines.append(f"State: {info['state']}")
        if "camera" in info:
            lines.append(f"Cam: {info['camera']}")
        if "target" in info:
            lines.append(f"Target: {info['target']}")
        if "vx" in info:
            lines.append(f"Vx: {info['vx']:+.3f}")
        if "vy" in info:
            lines.append(f"Vy: {info['vy']:+.3f}")
        if "vyaw" in info:
            lines.append(f"Vyaw: {info['vyaw']:+.3f}")
        if "fps" in info:
            lines.append(f"FPS: {info['fps']:.1f}")
        if "dt" in info:
            lines.append(f"dt: {info['dt']:.3f}s")

        if not lines:
            return frame

        bg_h = len(lines) * self._HUD_LINE_HEIGHT + self._HUD_PAD * 2
        bg_w = self._HUD_WIDTH

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (bg_w, bg_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        for i, line in enumerate(lines):
            y = self._HUD_PAD + (i + 1) * self._HUD_LINE_HEIGHT
            cv2.putText(
                frame, line, (self._HUD_PAD, y),
                self._HUD_FONT, self._HUD_FONT_SCALE,
                (255, 255, 255), self._HUD_FONT_THICKNESS, cv2.LINE_AA,
            )

        return frame


    def write_frame(self, frame: np.ndarray) -> None:

        self._frame_count += 1

        if not self._recording:
            return None

        # 延迟初始化 VideoWriter
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._writer = cv2.VideoWriter(
                self._record_path, fourcc, self._RECORD_FPS, (w, h),
            )

        self._writer.write(frame)


    def save_frame(self, frame: np.ndarray, tag: str = "") -> str | None:
        """
        Args:
            frame: 要保存的图像帧
            tag:   文件名后缀标签（如 'aligned'）

        Returns:
            保存的文件路径，未配置快照目录时返回 None。
        """
        if self._snapshot_dir is None:
            return None

        if tag:
            filename = f"frame_{self._frame_count:06d}_{tag}.png"
        else:
            filename = f"frame_{self._frame_count:06d}.png"

        filepath = os.path.join(self._snapshot_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath


    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._recording = False


    def __enter__(self) -> "DebugVisualizer":
        return self

    def __exit__(self, *args) -> None:
        self.release()
