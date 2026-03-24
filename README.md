# 🚁 基于视觉识别的微型物流无人机 — 地面站控制软件

> Ground Control Station (GCS) for Vision-Based Micro Logistics Drone

## 📋 项目简介

本项目为微型物流无人机系统的**地面站控制软件**部分，负责：
- **视觉流采集**：低延迟异步获取机载摄像头画面
- **目标检测**：基于 YOLOv8-OBB 的旋转目标检测与位姿估计
- **视觉伺服**：PD 控制律实现精确对准与闭环飞行控制
- **飞控通信**：MAVLink 协议封装，与 Pixhawk 飞控交互
- **任务调度**：有限状态机（FSM）统筹全局任务流转与容错保护

## 🏗️ 系统架构

```
地面站 PC（决策层）
  ├── YOLO-OBB 视觉推理    ← GPU 加速
  ├── PD 视觉伺服控制器
  └── FSM 全局状态机
        │
   WiFi 图传 / 数传链路
        │
  Pixhawk 飞控（控制层）
        │
  树莓派 Pico 2（执行层）
  ├── 双舵机夹爪驱动
  └── 红外传感器反馈
```

## 📂 项目结构

```
drone_gcs/
├── config/              # 参数配置
│   └── default.yaml
├── core/                # 五大核心模块
│   ├── streamer.py          # M1 - 低延迟视频流采集
│   ├── perception.py        # M2 - YOLO-OBB 推理
│   ├── servo_controller.py  # M3 - 视觉伺服控制
│   ├── mavlink_commander.py # M4 - MAVLink 通信
│   └── state_machine.py     # M5 - 全局状态机
├── utils/               # 通用工具库
├── tests/               # 单元 & 集成测试
├── scripts/             # 工具脚本
├── weights/             # YOLO 权重（不入库）
├── docs/                # 项目文档
├── main.py              # 主入口
└── requirements.txt     # 依赖清单
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建并激活 conda 环境
conda create -n drone_gcs python=3.10 -y
conda activate drone_gcs

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行测试

```bash
pytest tests/ -v
```

## 📖 文档

详细设计文档见 `docs/` 目录：
- `project_summary.md` — 项目逻辑总结
- `算法控制逻辑.md` — 算法控制模块设计
- `初期开发计划.md` — 开发计划与路线图

## 📄 License

MIT License
