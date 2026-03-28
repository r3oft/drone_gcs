import argparse
import os
import sys
import time
import select

import cv2
import numpy as np
import requests


def get_next_index(save_dir: str) -> int:
    """自动计算下一张图片的编号，避免覆盖已有文件。"""
    os.makedirs(save_dir, exist_ok=True)
    existing = [
        f for f in os.listdir(save_dir)
        if f.startswith("cargo_") and f.endswith(".jpg")
    ]
    if not existing:
        return 1
    indices = []
    for f in existing:
        stem = f.replace("cargo_", "").replace(".jpg", "")
        if stem.isdigit():
            indices.append(int(stem))
    return max(indices) + 1 if indices else 1


def capture_one_frame(host: str) -> np.ndarray | None:
    """从 /capture 端点获取一张 JPEG 静态图片并解码。"""
    url = f"http://{host}/capture"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img_array = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except requests.exceptions.RequestException as e:
        print(f"[!] 请求失败: {e}")
        return None


def save_frame(frame: np.ndarray, save_dir: str, idx: int) -> str:
    """保存帧为 JPEG 文件，返回文件路径。"""
    filename = os.path.join(save_dir, f"cargo_{idx:04d}.jpg")
    cv2.imwrite(filename, frame)
    return filename


# ────────────────────── 模式 1：自动批量采集 ───────────────────────────────

def run_auto_mode(host: str, save_dir: str, count: int, interval: float):
    """
    按照设定的间隔自动批量请求 /capture 端点。
    按 Ctrl+C 随时停止。
    """
    idx = get_next_index(save_dir)
    print(f"[*] 自动采集模式，目标: {count} 张，间隔: {interval}s")
    print(f"[*] 起始编号: cargo_{idx:04d}.jpg")
    print("[*] 按 Ctrl+C 随时停止\n")

    saved = 0
    while saved < count:
        try:
            frame = capture_one_frame(host)
            if frame is None:
                print("    解码失败，1s 后重试...")
                time.sleep(1)
                continue

            path = save_frame(frame, save_dir, idx)
            h, w = frame.shape[:2]
            print(f"[+] [{saved + 1}/{count}] {path}  ({w}x{h})")
            idx += 1
            saved += 1
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[*] 用户中断。")
            break

    print(f"\n[*] 共保存 {saved} 张图片到 {save_dir}")


# ────────────────────── 模式 2：手动逐张采集 ───────────────────────────────

def run_manual_mode(host: str, save_dir: str):
    """
    每按一次回车就抓拍一张。输入 'q' + 回车退出。
    适合需要仔细调整货箱角度的场景。
    """
    idx = get_next_index(save_dir)
    print(f"[*] 手动采集模式")
    print(f"[*] 起始编号: cargo_{idx:04d}.jpg")
    print("[*] 操作说明:")
    print("    - 按 [回车] 拍一张")
    print("    - 输入 q + [回车] 退出\n")

    # 先测试连接
    print("[*] 测试摄像头连接...", end=" ", flush=True)
    test = capture_one_frame(host)
    if test is None:
        print("失败！请检查摄像头是否在线。")
        sys.exit(1)
    h, w = test.shape[:2]
    print(f"成功！分辨率: {w}x{h}\n")

    saved = 0
    while True:
        try:
            user_input = input(f"[第 {saved + 1} 张] 按回车拍摄 (q=退出): ").strip().lower()
            if user_input == 'q':
                break

            frame = capture_one_frame(host)
            if frame is None:
                print("    抓取失败，请重试。")
                continue

            path = save_frame(frame, save_dir, idx)
            print(f"    ✓ 已保存: {path}")
            idx += 1
            saved += 1

        except KeyboardInterrupt:
            print("\n[*] 用户中断。")
            break

    print(f"\n[*] 共保存 {saved} 张图片到 {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="ESP32-S3 CameraWebServer 训练数据采集工具 (Headless)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--host", default="192.168.43.192",
        help="ESP32 的 IP 地址 (默认: 192.168.43.192)"
    )
    parser.add_argument(
        "--mode", choices=["auto", "manual"], default="manual",
        help=(
            "采集模式:\n"
            "  manual - 按回车手动逐张拍摄 (默认)\n"
            "  auto   - 自动按间隔批量拍摄"
        )
    )
    parser.add_argument(
        "--save_dir",
        default="datasets/cargo_obb/images/train",
        help="图片保存目录 (默认: datasets/cargo_obb/images/train)"
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="[auto 模式] 目标采集张数 (默认: 50)"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="[auto 模式] 两次采集间隔秒数 (默认: 1.0)"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  ESP32-S3 训练数据采集工具 (Headless)")
    print("=" * 55)
    print(f"  模式    : {args.mode}")
    print(f"  摄像头  : {args.host}")
    print(f"  保存目录: {args.save_dir}")
    print("=" * 55 + "\n")

    if args.mode == "manual":
        run_manual_mode(args.host, args.save_dir)
    else:
        run_auto_mode(args.host, args.save_dir, args.count, args.interval)


if __name__ == "__main__":
    main()
