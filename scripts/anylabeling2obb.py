import json
import os
import argparse
from pathlib import Path

def convert_to_yolo_obb(json_dir, output_dir, classes):
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_map = {name.strip(): i for i, name in enumerate(classes)}
    print(f"[*] 类别映射: {class_map}")
    
    count = 0
    for json_file in json_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
                
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            continue
            
        txt_file = output_dir / (json_file.stem + ".txt")
        lines = []
        
        for shape in data.get("shapes", []):
            # YOLO-OBB 只需要旋转框 (rotation) 或者多边形 (polygon 四点)
            if shape.get("shape_type") not in ["rotation", "polygon"]:
                continue
                
            label = shape.get("label", "")
            if not label or label not in class_map:
                continue
                
            points = shape.get("points", [])
            # 旋转框必须刚好 4 个点
            if len(points) != 4:
                continue
                
            class_id = class_map[label]
            
            # YOLO-OBB 的格式： class_id x1 y1 x2 y2 x3 y3 x4 y4 (全归一化为 0~1 的小数)
            normalized_points = []
            for p in points:
                # 处理可能越界的坐标点
                x = max(0.0, min(1.0, p[0] / img_w))
                y = max(0.0, min(1.0, p[1] / img_h))
                normalized_points.extend([x, y])
                
            line = f"{class_id} " + " ".join([f"{v:.6f}" for v in normalized_points])
            lines.append(line)
            
        # 即使没有找到对象，我们也会创建一个空文件，以防背景纯阴性样本报错
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        count += 1
            
    print(f"[+] 转换完成！共生成了 {count} 个 YOLO-OBB .txt 标注文件。")
    print(f"[+] 保存路径: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="AnyLabeling JSON 转换为 YOLO-OBB TXT")
    parser.add_argument("--json_dir", default="datasets/cargo_obb/images/train", help="包含 AnyLabeling JSON 文件的目录")
    parser.add_argument("--out_dir", default="datasets/cargo_obb/labels/train", help="输出 YOLO tx t文件的目录")
    # 允许传入多个 class，用逗号分隔，如: --classes cargo,landing_zone
    parser.add_argument("--classes", default="cargo,landing_zone", help="所有的类别名称，用逗号隔开，顺序对应 class_id 0, 1, 2...")
    
    args = parser.parse_args()
    classes = [c for c in args.classes.split(",") if c]
    convert_to_yolo_obb(args.json_dir, args.out_dir, classes)

if __name__ == "__main__":
    main()
