"""
AnyLabeling JSON → YOLO-OBB TXT 转换脚本

功能：
  - 将 AnyLabeling 导出的 rotation/polygon (4点) 标注转换为 YOLO-OBB 格式
  - 自动从 cargo_dataset.yaml 读取类别配置，确保类别 ID 一致
  - 支持标签别名映射（如 landing_zone → delivery_zone）
  - 处理 AnyLabeling 导出 JSON 中的 Infinity 等非标准值

用法：
  python scripts/anylabeling2obb.py
  python scripts/anylabeling2obb.py --json_dir datasets/cargo_obb/images/train \
                                     --out_dir datasets/cargo_obb/labels/train \
                                     --config config/cargo_dataset.yaml
"""

import json
import os
import re
import argparse
from pathlib import Path

import yaml

# ── 标签别名映射 ─────────────────────────────────────────────
# key = JSON 中实际标注的名称, value = 数据集配置中的正式类别名
LABEL_ALIASES = {
    "landing_zone": "delivery_zone",
}


def _sanitize_json(raw: str) -> str:
    """将 AnyLabeling 输出的非标准 JSON（含 Infinity / NaN）转换为合法 JSON。"""
    raw = re.sub(r'\bInfinity\b', '999999', raw)
    raw = re.sub(r'-Infinity\b', '-999999', raw)
    raw = re.sub(r'\bNaN\b', '0', raw)
    return raw


def load_classes_from_yaml(yaml_path: str) -> dict[str, int]:
    """从 cargo_dataset.yaml 加载类别映射 {类别名: 类别ID}。"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    # names 格式为 {0: "pickup_zone", 1: "delivery_zone"}
    return {v: k for k, v in names.items()}


def convert_to_yolo_obb(json_dir: str, output_dir: str, class_map: dict[str, int]):
    """
    遍历 json_dir 中的所有 .json 标注文件，转换为 YOLO-OBB 格式 .txt 写入 output_dir。

    YOLO-OBB 行格式：class_id x1 y1 x2 y2 x3 y3 x4 y4  （坐标归一化到 [0,1]）
    """
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] 类别映射: {class_map}")
    print(f"[*] 标签别名: {LABEL_ALIASES}")
    print(f"[*] 源目录:   {json_dir}")
    print(f"[*] 输出目录: {output_dir}")
    print()

    converted = 0
    skipped_no_size = 0
    skipped_shapes = 0
    label_counts: dict[str, int] = {}
    unknown_labels: dict[str, int] = {}

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print("[!] 未找到任何 .json 文件，请检查路径。")
        return

    for json_file in json_files:
        raw = json_file.read_text(encoding="utf-8")
        try:
            data = json.loads(_sanitize_json(raw))
        except json.JSONDecodeError as e:
            print(f"[!] JSON 解析失败，跳过: {json_file.name} ({e})")
            continue

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            skipped_no_size += 1
            continue

        lines: list[str] = []

        for shape in data.get("shapes", []):
            # YOLO-OBB 只接受旋转框 (rotation) 或四点多边形 (polygon)
            if shape.get("shape_type") not in ("rotation", "polygon"):
                skipped_shapes += 1
                continue

            # ── 标签处理 ──
            label = (shape.get("label") or "").strip()
            if not label:
                skipped_shapes += 1
                continue

            # 应用别名映射
            label = LABEL_ALIASES.get(label, label)

            if label not in class_map:
                unknown_labels[label] = unknown_labels.get(label, 0) + 1
                continue

            points = shape.get("points", [])
            if len(points) != 4:
                skipped_shapes += 1
                continue

            class_id = class_map[label]
            label_counts[label] = label_counts.get(label, 0) + 1

            # 坐标归一化并裁剪到 [0, 1]
            normalized: list[float] = []
            for p in points:
                x = max(0.0, min(1.0, p[0] / img_w))
                y = max(0.0, min(1.0, p[1] / img_h))
                normalized.extend([x, y])

            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in normalized)
            lines.append(line)

        # 即使没有找到有效标注，也写入空文件（负样本/背景图不会导致训练报错）
        txt_file = output_dir / (json_file.stem + ".txt")
        txt_file.write_text("\n".join(lines), encoding="utf-8")
        converted += 1

    # ── 统计汇总 ──
    print(f"[+] 转换完成！共处理 {converted} 个 JSON ➜ TXT")
    print(f"    各类别目标数量: {label_counts}")
    if unknown_labels:
        print(f"[!] 未识别的标签 (已跳过): {unknown_labels}")
    if skipped_no_size:
        print(f"[!] 缺少 imageWidth/Height 而跳过的文件: {skipped_no_size}")
    if skipped_shapes:
        print(f"[!] 跳过的非 rotation/polygon 或无效标注: {skipped_shapes}")
    print(f"[+] 输出路径: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="AnyLabeling JSON → YOLO-OBB TXT 转换工具"
    )
    parser.add_argument(
        "--json_dir",
        default="datasets/cargo_obb/images/train",
        help="包含 AnyLabeling JSON 文件的目录",
    )
    parser.add_argument(
        "--out_dir",
        default="datasets/cargo_obb/labels/train",
        help="输出 YOLO-OBB .txt 文件的目录",
    )
    parser.add_argument(
        "--config",
        default="config/cargo_dataset.yaml",
        help="数据集 YAML 配置文件路径（用于读取类别列表）",
    )
    args = parser.parse_args()

    class_map = load_classes_from_yaml(args.config)
    convert_to_yolo_obb(args.json_dir, args.out_dir, class_map)


if __name__ == "__main__":
    main()
