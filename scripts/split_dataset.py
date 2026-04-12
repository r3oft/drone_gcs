"""
数据集清洗、转换与 Train/Val 划分脚本

最终目录结构：
  datasets/cargo_obb/
  ├── images/
  │   ├── train/   (原始 json + jpg 保持不动)
  │   └── val/     (划分出来的验证集 jpg)
  └── labels/
      ├── train/   (训练集的 .txt)
      └── val/     (验证集的 .txt)
"""

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import yaml

LABEL_ALIASES = {
    "landing_zone": "delivery_zone",
}


def _sanitize_json(raw: str) -> str:
    """将 AnyLabeling 输出的非标准 JSON（含 Infinity / NaN）转换为合法 JSON。"""
    raw = re.sub(r'\bInfinity\b', '999999', raw)
    raw = re.sub(r'-Infinity\b', '-999999', raw)
    raw = re.sub(r'\bNaN\b', '0', raw)
    return raw


def detect_class(json_path: Path) -> str | None:
    raw = json_path.read_text(encoding="utf-8")
    try:
        data = json.loads(_sanitize_json(raw))
    except json.JSONDecodeError:
        return None

    for shape in data.get("shapes", []):
        if shape.get("shape_type") not in ("rotation", "polygon"):
            continue
        label = (shape.get("label") or "").strip()
        if not label:
            continue
        label = LABEL_ALIASES.get(label, label)
        return label

    return None


def load_yaml_config(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_conversion(src_dir: str, tmp_label_dir: str, config_path: str):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from anylabeling2obb import convert_to_yolo_obb, load_classes_from_yaml

    class_map = load_classes_from_yaml(config_path)
    convert_to_yolo_obb(src_dir, tmp_label_dir, class_map)


def split_and_organize(
    src_image_dir: Path,
    tmp_label_dir: Path,
    dataset_root: Path,
    val_ratio: float,
    seed: int,
):
    # ── 1. 按类别分组 ──
    class_groups: dict[str, list[str]] = {}  # {class_name: [stem1, stem2, ...]}
    unlabeled: list[str] = []

    json_files = sorted(src_image_dir.glob("*.json"))
    for jf in json_files:
        img_path = src_image_dir / (jf.stem + ".jpg")
        if not img_path.exists():
            continue
        cls = detect_class(jf)
        if cls is None:
            unlabeled.append(jf.stem)
        else:
            class_groups.setdefault(cls, []).append(jf.stem)

    print("\n[*] 数据集类别分布：")
    for cls, stems in sorted(class_groups.items()):
        print(f"    {cls}: {len(stems)} 张")
    if unlabeled:
        print(f"    (无有效标注): {len(unlabeled)} 张")

    # ── 2. 分层随机划分 ──
    rng = random.Random(seed)
    train_stems: list[str] = []
    val_stems: list[str] = []

    for cls, stems in class_groups.items():
        rng.shuffle(stems)
        n_val = max(1, int(len(stems) * val_ratio))  # 每类至少 1 张 val
        val_stems.extend(stems[:n_val])
        train_stems.extend(stems[n_val:])

    # 无标注的图片全部放入训练集（作为背景负样本）
    train_stems.extend(unlabeled)

    print(f"\n[*] 划分结果 (val_ratio={val_ratio}, seed={seed})：")
    print(f"    训练集: {len(train_stems)} 张")
    print(f"    验证集: {len(val_stems)} 张")

    dirs = {
        "img_train": dataset_root / "images" / "train",
        "img_val":   dataset_root / "images" / "val",
        "lbl_train": dataset_root / "labels" / "train",
        "lbl_val":   dataset_root / "labels" / "val",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def copy_pair(stem: str, img_dest: Path, lbl_dest: Path):
        img_src = src_image_dir / (stem + ".jpg")
        lbl_src = tmp_label_dir / (stem + ".txt")

        if img_src.exists() and img_src.resolve() != (img_dest / img_src.name).resolve():
            shutil.copy2(img_src, img_dest / img_src.name)
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dest / lbl_src.name)

    # 训练集标注
    for stem in train_stems:
        copy_pair(stem, dirs["img_train"], dirs["lbl_train"])

    # 验证集
    for stem in val_stems:
        copy_pair(stem, dirs["img_val"], dirs["lbl_val"])

    train_imgs = len(list(dirs["img_train"].glob("*.jpg")))
    train_lbls = len(list(dirs["lbl_train"].glob("*.txt")))
    val_imgs = len(list(dirs["img_val"].glob("*.jpg")))
    val_lbls = len(list(dirs["lbl_val"].glob("*.txt")))

    print(f"\n[+] 文件组织完成！")
    print(f"    images/train: {train_imgs} 张图片")
    print(f"    labels/train: {train_lbls} 个标注")
    print(f"    images/val:   {val_imgs} 张图片")
    print(f"    labels/val:   {val_lbls} 个标注")

    if train_imgs != train_lbls:
        print(f"[!] 警告：训练集图片数({train_imgs}) ≠ 标注数({train_lbls})")
    if val_imgs != val_lbls:
        print(f"[!] 警告：验证集图片数({val_imgs}) ≠ 标注数({val_lbls})")

    print(f"\n[+] 数据集已准备就绪，可使用以下命令开始训练：")
    print(f"    python scripts/train_obb.py")


def main():
    parser = argparse.ArgumentParser(
        description="数据集清洗、转换与 Train/Val 划分"
    )
    parser.add_argument(
        "--src_dir",
        default="datasets/cargo_obb/images/train",
        help="原始 AnyLabeling JSON + 图片所在目录",
    )
    parser.add_argument(
        "--config",
        default="config/cargo_dataset.yaml",
        help="数据集 YAML 配置文件路径",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="验证集占比 (默认 0.2，即 20%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（可复现的划分）",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="运行前清空已有的 labels/ 和 images/val/ 目录",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    cfg = load_yaml_config(args.config)

    # 从 yaml 的 path 字段推断 dataset_root
    # cargo_dataset.yaml 中 path 是相对于 config/ 的 ../datasets/cargo_obb
    config_parent = Path(args.config).resolve().parent
    dataset_root = (config_parent / cfg["path"]).resolve()
    print(f"[*] 数据集根目录: {dataset_root}")

    tmp_label_dir = dataset_root / "labels" / "_all"

    if args.clean:
        print("[*] 清理旧数据...")
        for d in [
            dataset_root / "labels",
            dataset_root / "images" / "val",
        ]:
            if d.exists():
                shutil.rmtree(d)
                print(f"    已删除: {d}")

    print("\n" + "=" * 60)
    print("  Step 1: AnyLabeling JSON → YOLO-OBB TXT 转换")
    print("=" * 60)
    run_conversion(str(src_dir), str(tmp_label_dir), args.config)

    print("\n" + "=" * 60)
    print("  Step 2: 分层划分 & 文件组织")
    print("=" * 60)
    split_and_organize(src_dir, tmp_label_dir, dataset_root, args.val_ratio, args.seed)

    if tmp_label_dir.exists():
        shutil.rmtree(tmp_label_dir)
        print(f"\n[*] 已清理临时目录: {tmp_label_dir}")


if __name__ == "__main__":
    main()
