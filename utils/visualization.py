import math
import os

import cv2
import numpy as np


CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "pickup_zone":  (50, 205, 50),   # 浅绿色
    "delivery_zone": (255, 180, 0),  # 亮蓝色
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

    _HUD_LINE_HEIGHT = 14
    _HUD_PAD = 5
    _HUD_WIDTH = 180
    _HUD_MAX_WIDTH_RATIO = 0.72
    _HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
    _HUD_FONT_SCALE = 0.38
    _HUD_MIN_FONT_SCALE = 0.30
    _HUD_FONT_THICKNESS = 1
    _LABEL_FONT_SCALE = 0.38
    _LABEL_PAD = 2
    _RECORD_FPS = 15

    def __init__(
        self,
        record_path: str | None = None,
        snapshot_dir: str | None = None,
        hud_corner: str = "top_left",
        record_segment_s: float = 0.0,
    ) -> None:
        """
        Args:
            record_path:  视频录制输出路径（.avi 格式）。
                          为 None 时不录制。
            snapshot_dir: 帧快照保存目录。
                          为 None 时不支持 save_frame()。
            hud_corner:   HUD 显示位置：
                          top_left / top_right / bottom_left / bottom_right。
            record_segment_s: 分段录制时长（秒）。为 0 时保持单文件录制。
        """
        valid_corners = {"top_left", "top_right", "bottom_left", "bottom_right"}
        if hud_corner not in valid_corners:
            raise ValueError(f"hud_corner 必须是 {valid_corners} 之一，收到: {hud_corner}")
        if record_segment_s < 0:
            raise ValueError(f"record_segment_s 必须 >= 0，收到: {record_segment_s}")

        # ── 视频录制 ──
        self._record_path = record_path
        self._writer: cv2.VideoWriter | None = None  # 延迟初始化
        self._recording = record_path is not None
        self._hud_corner = hud_corner
        self._record_segment_s = float(record_segment_s)
        self._segment_frame_limit = (
            max(1, int(round(self._RECORD_FPS * self._record_segment_s)))
            if self._record_segment_s > 0 else 0
        )
        self._segment_index = 0
        self._segment_start_frame = 0
        self._current_record_path: str | None = None

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
        thickness: int = 2,
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
        cv2.drawContours(frame, [box], 0, color, thickness)

        # 绘制中心点
        cv2.circle(frame, (int(u), int(v)), max(4, thickness + 2), color, -1)

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
                text, self._HUD_FONT, self._LABEL_FONT_SCALE, self._HUD_FONT_THICKNESS,
            )
            # 文字背景矩形
            cv2.rectangle(
                frame,
                (text_x, text_y - th_text - self._LABEL_PAD * 2),
                (text_x + tw + self._LABEL_PAD * 2, text_y),
                color, -1,
            )
            # 文字（黑色，在彩色背景上）
            cv2.putText(
                frame, text, (text_x + self._LABEL_PAD, text_y - self._LABEL_PAD),
                self._HUD_FONT, self._LABEL_FONT_SCALE,
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
        state_line = []
        if "state" in info:
            state_line.append(str(info["state"]))
        if "mode" in info:
            state_line.append(str(info["mode"]))
        if state_line:
            lines.append(" | ".join(state_line))
        if "camera" in info:
            lines.append(f"Cam {info['camera']}")
        target_line = []
        if "target" in info:
            target_line.append(f"T {info['target']}")
        if "conf" in info:
            target_line.append(f"{info['conf']:.0%}")
        if target_line:
            lines.append(" ".join(target_line))
        target_debugs = info.get("target_debugs", [])
        if all(k in info for k in ("vx", "vy", "vyaw")) and not target_debugs:
            lines.append(
                f"V {info['vx']:+.3f} {info['vy']:+.3f} {info['vyaw']:+.3f}"
            )
        if all(k in info for k in ("err_x", "err_y", "err_yaw")) and not target_debugs:
            lines.append(
                f"E {info['err_x']:+.1f} {info['err_y']:+.1f} {info['err_yaw']:+.2f}"
            )
        if all(k in info for k in ("p_x", "p_y", "p_yaw")) and not target_debugs:
            lines.append(
                f"P {info['p_x']:+.3f} {info['p_y']:+.3f} {info['p_yaw']:+.3f}"
            )
        if all(k in info for k in ("d_x", "d_y", "d_yaw")) and not target_debugs:
            lines.append(
                f"D {info['d_x']:+.3f} {info['d_y']:+.3f} {info['d_yaw']:+.3f}"
            )
        for target_debug in target_debugs:
            label = str(target_debug.get("label", "target"))
            if label == "pickup_zone":
                short_label = "PICK"
            elif label == "delivery_zone":
                short_label = "DELV"
            else:
                short_label = label[:4].upper()
            active_mark = "*" if target_debug.get("active") else " "
            prefix = f"{active_mark}{short_label}"
            if all(k in target_debug for k in ("conf", "vx", "vy", "vyaw")):
                lines.append(
                    f"{prefix} {target_debug['conf']:.0%} V "
                    f"{target_debug['vx']:+.3f} {target_debug['vy']:+.3f} "
                    f"{target_debug['vyaw']:+.3f}"
                )
            if all(k in target_debug for k in ("err_x", "err_y", "err_yaw")):
                lines.append(
                    f"{prefix} E {target_debug['err_x']:+.1f} "
                    f"{target_debug['err_y']:+.1f} {target_debug['err_yaw']:+.2f}"
                )
        fps_line = []
        if "fps" in info:
            fps_line.append(f"FPS {info['fps']:.1f}")
        if "source_fps" in info:
            fps_line.append(f"Src {info['source_fps']:.1f}")
        if fps_line:
            lines.append(" ".join(fps_line))
        timing_line = []
        if "read_fps" in info:
            timing_line.append(f"Read {info['read_fps']:.1f}")
        if "dt" in info:
            timing_line.append(f"dt {info['dt']:.3f}s")
        if timing_line:
            lines.append(" ".join(timing_line))

        if not lines:
            return frame

        max_bg_w = max(1, int(frame.shape[1] * self._HUD_MAX_WIDTH_RATIO))
        font_scale = self._HUD_FONT_SCALE

        while True:
            text_sizes = [
                cv2.getTextSize(
                    line, self._HUD_FONT, font_scale, self._HUD_FONT_THICKNESS,
                )[0]
                for line in lines
            ]
            max_text_w = max(width for width, _ in text_sizes)
            if (
                max_text_w + self._HUD_PAD * 2 <= max_bg_w
                or font_scale <= self._HUD_MIN_FONT_SCALE
            ):
                break
            font_scale = max(self._HUD_MIN_FONT_SCALE, font_scale - 0.02)

        text_h = max(height for _, height in text_sizes)
        line_height = max(self._HUD_LINE_HEIGHT, text_h + 4)
        hud_x = self._HUD_PAD
        if self._hud_corner.endswith("right"):
            hud_x = max(self._HUD_PAD, frame.shape[1] - max_text_w - self._HUD_PAD)

        for i, line in enumerate(lines):
            if self._hud_corner.startswith("bottom"):
                y = frame.shape[0] - self._HUD_PAD - (len(lines) - 1 - i) * line_height
            else:
                y = self._HUD_PAD + text_h + i * line_height
            if y >= frame.shape[0]:
                break
            # Black stroke keeps the compact HUD readable without covering the
            # video with a large opaque panel.
            cv2.putText(
                frame, line, (hud_x, y),
                self._HUD_FONT, font_scale,
                (0, 0, 0), self._HUD_FONT_THICKNESS + 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, line, (hud_x, y),
                self._HUD_FONT, font_scale,
                (255, 255, 255), self._HUD_FONT_THICKNESS, cv2.LINE_AA,
            )

        return frame


    def write_frame(self, frame: np.ndarray) -> None:

        self._frame_count += 1

        if not self._recording:
            return None

        if self._should_rotate_segment():
            self._close_writer()

        if self._writer is None:
            self._open_writer(frame)

        self._writer.write(frame)

    def flush_recording(self) -> None:
        """
        关闭当前视频片段但保持录制状态。

        用于长时间运行时的断流保护：当前片段被 release 后即可被播放器读取；
        后续恢复来帧时会自动打开下一段文件。
        """
        if self._record_segment_s <= 0:
            return
        self._close_writer()

    def _should_rotate_segment(self) -> bool:
        if self._writer is None or self._segment_frame_limit <= 0:
            return False
        written_in_segment = self._frame_count - self._segment_start_frame
        return written_in_segment >= self._segment_frame_limit

    def _next_record_path(self) -> str:
        if self._record_path is None:
            raise RuntimeError("record_path 未设置，无法录制")

        if self._record_segment_s <= 0:
            return self._record_path

        root, ext = os.path.splitext(self._record_path)
        ext = ext or ".avi"
        self._segment_index += 1
        return f"{root}_seg{self._segment_index:04d}{ext}"

    def _open_writer(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        record_path = self._next_record_path()
        record_dir = os.path.dirname(record_path)
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(record_path, fourcc, self._RECORD_FPS, (w, h))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"无法打开视频录制文件: {record_path}")

        self._writer = writer
        self._current_record_path = record_path
        self._segment_start_frame = self._frame_count

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._current_record_path = None


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
        self._close_writer()
        self._recording = False


    def __enter__(self) -> "DebugVisualizer":
        return self

    def __exit__(self, *args) -> None:
        self.release()
