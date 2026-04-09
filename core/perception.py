import os

import numpy as np

from ultralytics import YOLO

from utils.geometry import normalize_obb_angle


class TargetPoseEstimator:
    """
    YOLOv8-OBB 旋转目标检测与位姿特征提取（M2 模块）。

    支持双类别检测（pickup_zone / delivery_zone），
    由调用方通过 target_cls_id 参数指定当前检测目标。
    """

    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.6,
        device: str = "cuda:0",
    ) -> None:
        """
        加载 YOLO-OBB 模型并执行 GPU 预热推理。

        Args:
            weights_path:   YOLO-OBB 权重文件路径（.pt 格式）
            conf_threshold: NMS 置信度下限阈值（低于此值的检测将被丢弃）
            device:         推理设备

        Raises:
            FileNotFoundError: 权重文件不存在
        """
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"权重文件不存在: {weights_path}"
            )

        self._conf_threshold = conf_threshold
        self._device = device

        self._model = YOLO(weights_path)

        warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(warmup_frame, verbose=False, device=self._device)

    def process_frame(
        self,
        frame: np.ndarray,
        target_cls_id: int,
    ) -> dict | None:
        """
        Args:
            frame:          BGR 图像帧（H×W×3，np.uint8）
            target_cls_id:  目标类别 ID（0=pickup_zone, 1=delivery_zone）

        Returns:
            位姿字典 {"u", "v", "theta", "conf", "w", "h"}，
            或 None（未检测到符合条件的目标）。
            theta 已经过 C₂ 对称归一化，范围 [-π/4, π/4]。
        """
        results = self._model(frame, verbose=False, device=self._device)
        obb = results[0].obb

        if obb is None or len(obb) == 0:
            return None

        xywhr = obb.xywhr.cpu().numpy()    # [N, 5]: cx, cy, w, h, rotation_rad
        confs = obb.conf.cpu().numpy()      # [N]
        classes = obb.cls.cpu().numpy()     # [N]

        cls_mask = classes == target_cls_id
        if not np.any(cls_mask):
            return None

        xywhr = xywhr[cls_mask]
        confs = confs[cls_mask]

        conf_mask = confs >= self._conf_threshold
        if not np.any(conf_mask):
            return None

        xywhr = xywhr[conf_mask]
        confs = confs[conf_mask]

        best_idx = np.argmax(confs)
        cx, cy, w, h, theta_raw = xywhr[best_idx]
        best_conf = float(confs[best_idx])
        theta = normalize_obb_angle(float(theta_raw), symmetry_order=2)

        return {
            "u":     float(cx),
            "v":     float(cy),
            "theta": theta,
            "conf":  best_conf,
            "w":     float(w),
            "h":     float(h),
        }
