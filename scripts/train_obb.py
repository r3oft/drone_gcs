"""
YOLOv8n-OBB 训练脚本
"""

import os
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    print(f"[*] 项目根目录: {PROJECT_ROOT}")
    print(f"[*] 当前工作目录: {os.getcwd()}")

    model = YOLO('yolov8n-obb.pt')  

    results = model.train(
        data=str(PROJECT_ROOT / 'config' / 'cargo_dataset.yaml'),
        epochs=150,
        imgsz=640,
        batch=32,
        device='0',
        project=str(PROJECT_ROOT / 'weights'),
        name='cargo_obb_run',
        cache=True,
        workers=4,
        patience=30,
        warmup_epochs=3,
        lr0=0.01,
        amp=True,
    )

    save_dir = Path(results.save_dir)
    best_weight = save_dir / 'weights' / 'best.pt'
    print("\n" + "=" * 60)
    print("[+] 训练完成！")
    print(f"    最佳权重: {best_weight}")
    print(f"    结果目录: {save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
