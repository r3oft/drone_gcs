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
        targets = self.process_frame_all(frame, target_cls_ids=[target_cls_id])
        target = targets.get(int(target_cls_id))
        if target is None:
            return None

        return {
            "u":     target["u"],
            "v":     target["v"],
            "theta": target["theta"],
            "conf":  target["conf"],
            "w":     target["w"],
            "h":     target["h"],
        }

    def process_frame_all(
        self,
        frame: np.ndarray,
        target_cls_ids: list[int] | tuple[int, ...] | set[int] | None = None,
    ) -> dict[int, dict]:
        """
        单次 YOLO 推理中提取多个类别的最高置信 OBB 目标。

        Args:
            frame:          BGR 图像帧（H×W×3，np.uint8）
            target_cls_ids: 需要保留的类别 ID 集合。为 None 时返回所有类别。

        Returns:
            {cls_id: pose_dict}。每个类别最多返回一个最高置信目标。
            pose_dict 包含 {"cls_id", "u", "v", "theta", "conf", "w", "h"}。
        """
        results = self._model(frame, verbose=False, device=self._device)
        obb = results[0].obb

        if obb is None or len(obb) == 0:
            return {}

        xywhr = obb.xywhr.cpu().numpy()    # [N, 5]: cx, cy, w, h, rotation_rad
        confs = obb.conf.cpu().numpy()      # [N]
        classes = obb.cls.cpu().numpy().astype(int)  # [N]

        allowed_cls_ids = (
            {int(cls_id) for cls_id in target_cls_ids}
            if target_cls_ids is not None else None
        )
        targets: dict[int, dict] = {}

        for cls_id in sorted(set(classes.tolist())):
            if allowed_cls_ids is not None and cls_id not in allowed_cls_ids:
                continue

            mask = (classes == cls_id) & (confs >= self._conf_threshold)
            if not np.any(mask):
                continue

            cls_xywhr = xywhr[mask]
            cls_confs = confs[mask]
            best_idx = int(np.argmax(cls_confs))
            cx, cy, w, h, theta_raw = cls_xywhr[best_idx]
            best_conf = float(cls_confs[best_idx])
            theta = normalize_obb_angle(float(theta_raw), symmetry_order=2)

            targets[cls_id] = {
                "cls_id": cls_id,
                "u":      float(cx),
                "v":      float(cy),
                "theta":  theta,
                "conf":   best_conf,
                "w":      float(w),
                "h":      float(h),
            }

        return targets
