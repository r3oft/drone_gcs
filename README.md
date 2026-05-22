# 基于视觉识别的微型物流无人机地面站控制软件

Ground Control Station for a Vision-Based Micro Logistics Drone

## 项目概述

本项目为UCAS EE专业电子系统设计课程的课程设计。该项目面向室内固定场景下的微型物流无人机任务，提供运行在地面站 PC 上的视觉感知、闭环控制、飞控通信与任务调度软件。系统以机载 ESP32-S3 摄像头图像为主要观测输入，通过 YOLO-OBB 识别取货区与投递区，利用视觉伺服控制律生成机体系速度指令，并经 MAVLink 与 Pixhawk 飞控交互，完成起飞、对准、下降、取货、转运、投递与降落等任务阶段。


## 主要功能

### 视觉流采集

`core/streamer.py` 实现 `ZeroLatencyStreamer`，用于从 ESP32-S3 CameraWebServer 获取实时图像。采集器优先连接 MJPEG `/stream`，当流不可用或被占用时回退到 `/capture` 抓拍轮询，并通过后台线程缓存最新帧，降低主控制循环中的 I/O 阻塞。

### 目标检测与位姿提取

`core/perception.py` 封装 Ultralytics YOLO-OBB 推理流程。模型输出旋转目标框后，程序提取目标中心、朝向、置信度和框尺寸，并按任务阶段选择目标类别：

- `pickup_zone`: 取货区，默认类别 ID 为 `0`；
- `delivery_zone`: 投递区，默认类别 ID 为 `1`。

### 视觉伺服控制

`core/servo_controller.py` 将图像平面误差转换为机体系速度指令 `(vx, vy, yaw_rate)`。控制器支持比例-微分控制、死区、速度限幅、图像轴到机体系轴映射和符号修正，便于适配外置摄像头的实际安装方向。

### 飞控通信

`core/flight_bridge.py` 基于 DroneKit 与 pymavlink 实现 Pixhawk 通信，负责连接、心跳监测、解锁起飞、降落、模式切换、遥测读取和机体系速度指令下发。当前主线实现以 `FlightBridge` 为飞控抽象，不再使用早期文档中的独立 `MavlinkCommander` 模块。

### 末端执行器通信

`core/mcu_bridge.py` 提供 PC 直连 MCU 串口的默认实现，用于向末端执行器发送 `RESET`、`START_GRAB`、`START_RELEASE` 等文本命令，并等待 `RESET_DONE`、`GRAB_DONE`、`RELEASE_DONE` 等反馈。`core/flight_bridge.py` 中仍保留经 Pixhawk `SERIAL_CONTROL` 透传的兼容路径。

### 全局任务状态机

`core/state_machine.py` 实现全局有限状态机，将视觉、飞控和 MCU 动作组织为完整任务流程：

```text
IDLE -> RESET -> INBOUND
     -> TASK_REC_ALIGN -> TASK_REC_DESCEND -> TASK_REC_WAIT_LOAD
     -> TRANS_DELIVERY
     -> TASK_REL_ALIGN -> TASK_REL_DESCEND -> TASK_REL_RELEASE
     -> OUTBOUND -> IDLE
```

当出现飞控心跳异常、外部强制切入 RTL/LAND、MCU 响应超时或视觉链路长时间不可用等情况时，状态机会进入 `EMERGENCY` 并请求安全返航或降落。

## 系统结构

```text
drone_gcs/
├── main.py                     # 主控制入口，负责组装运行组件并执行主循环
├── config/
│   ├── default.yaml            # 主程序默认配置
│   └── cargo_dataset.yaml      # YOLO-OBB 数据集配置
├── core/
│   ├── interfaces.py           # 飞控与 MCU 抽象接口
│   ├── streamer.py             # ESP32-S3 低延迟图像采集
│   ├── perception.py           # YOLO-OBB 目标检测与位姿估计
│   ├── servo_controller.py     # 视觉伺服 PD 控制器
│   ├── flight_bridge.py        # DroneKit / MAVLink 飞控桥接
│   ├── mcu_bridge.py           # PC 直连 MCU 串口桥接
│   └── state_machine.py        # 全局任务有限状态机
├── utils/
│   ├── config_manager.py       # YAML 配置读取与命令行覆盖
│   ├── geometry.py             # 图像误差与角度归一化工具
│   ├── logger.py               # 日志与飞行黑匣子记录
│   ├── mock.py                 # 无硬件联调用 mock 组件
│   ├── perf_monitor.py         # 主循环性能统计
│   └── visualization.py        # HUD 与 OBB 可视化工具
├── scripts/
│   ├── monitor_drone.py        # 飞控遥测监控脚本
│   ├── run_visual_hud.py       # 实时视觉 HUD 与录像工具
│   ├── collect_train_data.py   # ESP32-S3 图像采集工具
│   ├── anylabeling2obb.py      # AnyLabeling 标注转 YOLO-OBB
│   ├── split_dataset.py        # 数据清洗、转换与训练/验证划分
│   ├── train_obb.py            # YOLO-OBB 训练入口
│   └── test_streamer_live.py   # 视觉流实机连通性测试
├── tests/                      # 单元测试与集成测试
├── docs/                       # 设计说明、调试流程与飞控配置文档
├── dronekit/                   # vendored DroneKit 源码与兼容 shim
├── requirements.txt            # Python 依赖
├── pytest.ini                  # pytest 默认配置
└── README.md
```

## 环境配置

### 基础环境

推荐在 WSL2 Ubuntu 或原生 Linux 环境中运行。开发与实测主要围绕 Python 3.10、Conda 环境和 Pixhawk/ESP32-S3/MCU 外设展开。

```bash
conda create -n drone_gcs python=3.10 -y
conda activate drone_gcs
```

### 安装 Python 依赖

```bash
pip install -r requirements.txt
```

说明：

- `requirements.txt` 包含 Ultralytics、PyTorch、OpenCV、pymavlink、pyserial、pytest 等主依赖；
- `dronekit/` 目录中保留了项目使用的 DroneKit 源码，建议以 editable 模式安装，避免本地源码目录与 Python 包导入路径冲突；
- 若需要实时显示 HUD，OpenCV 需为带 GUI 支持的 `opencv-python`，不应使用仅 headless 的构建；
- 若使用 GPU 推理，请确认 PyTorch 与本机 CUDA 版本匹配。CPU 调试可在运行时通过 `--device cpu` 覆盖。

### 关键硬件与链路

实机运行前需要确认以下链路可用：

- ESP32-S3 CameraWebServer 可通过 `http://<host>:81/stream` 或 `http://<host>/capture` 访问；
- Pixhawk MAVLink 链路可通过 UDP 或串口访问，例如 `udp:192.168.4.1:14550` 或 `/dev/ttyUSB0`；
- MCU 末端执行器默认通过 PC 直连串口访问，例如 `/dev/ttyACM0`；
- WSL2 环境下使用 USB 串口设备时，需要先通过 `usbipd` 将设备附加到 WSL，再确认 `/dev/ttyUSB*` 或 `/dev/ttyACM*` 存在。

默认链路参数集中在 `config/default.yaml` 中，常用字段包括：

- `stream.*`: 摄像头地址、超时和 `/stream`/`/capture` 回退策略；
- `perception.*`: 模型权重、置信度阈值和推理设备；
- `servo.*`: 取货/投递阶段的控制增益、死区、限幅和轴映射；
- `mavlink.*`: Pixhawk 连接串、波特率、心跳超时和重连策略；
- `mcu.*`: MCU 串口、波特率、通信方式和动作超时；
- `mission.*`: 任务剖面、最大飞行高度、无 MCU 飞测参数；
- `logging.*`: 日志目录和飞行黑匣子开关。

## 主程序运行说明

### 1. 无硬件 wiring check

仅检查主程序组件装配，不启动完整任务：

```bash
python main.py --mode mock --no-flight-recorder --perf-print-interval 0
```

### 2. mock 模式短时闭环联调

用于在无飞控、无摄像头、无 MCU 的桌面环境中验证主循环、状态机和接口装配：

```bash
python main.py \
  --mode mock \
  --start \
  --mock-fast \
  --duration 2 \
  --no-flight-recorder \
  --perf-print-interval 0
```

### 3. live 模式实机运行

实机任务需要显式确认启动，避免误触发飞行流程：

```bash
python main.py \
  --mode live \
  --start \
  --confirm-live-start \
  --conn udp:192.168.4.1:14550 \
  --host 192.168.43.192 \
  --mcu-port /dev/ttyACM0
```

常用覆盖参数：

- `--config`: 指定 YAML 配置文件；
- `--weights`: 指定 YOLO-OBB 权重；
- `--device`: 指定推理设备，如 `cuda:0` 或 `cpu`；
- `--conn` / `--baud`: 覆盖 Pixhawk 连接串与串口波特率；
- `--host` / `--stream-url` / `--capture-url`: 覆盖摄像头地址；
- `--prefer-capture`: 强制优先使用 `/capture`；
- `--mcu-port` / `--mcu-baud`: 覆盖 MCU 串口；
- `--mcu-transport`: 在 `direct_serial` 与 `pixhawk_serial_control` 间切换；
- `--mission-profile no_mcu_flight_test`: 使用无 MCU 飞行测试剖面；
- `--debug-hud` 或 `--debug-hud-record-path`: 打开主循环 HUD 显示或录像；
- `--status-log-interval`: 周期性输出飞行状态、视觉帧率和控制量。

## 常用辅助程序

### 飞控遥测监控

`scripts/monitor_drone.py` 是当前推荐的飞控状态监控入口，可用于单独检查 Pixhawk 连接、模式、解锁状态、高度、电池与心跳。

```bash
python scripts/monitor_drone.py --conn /dev/ttyUSB0 --baud 57600
```

如需保存监控数据：

```bash
python scripts/monitor_drone.py \
  --conn /dev/ttyUSB0 \
  --baud 57600 \
  --log-csv logs/flight_monitor.csv
```

### 实时视觉 HUD

用于单独验证摄像头、YOLO-OBB、目标框绘制和视觉伺服控制量：

```bash
python scripts/run_visual_hud.py \
  --target pickup_zone \
  --host 192.168.43.192 \
  --weights weights/cargo_obb_run2/weights/best.pt \
  --device cuda:0 \
  --display
```

仅录像、不显示窗口时：

```bash
python scripts/run_visual_hud.py \
  --target delivery_zone \
  --record-path logs/visual_hud.avi \
  --duration 60
```

### 数据采集、转换与训练

采集 ESP32-S3 图像：

```bash
python scripts/collect_train_data.py --host 192.168.43.192 --mode manual
```

将 AnyLabeling 标注转换为 YOLO-OBB 并划分训练/验证集：

```bash
python scripts/split_dataset.py --clean
```

训练 YOLO-OBB 模型：

```bash
python scripts/train_obb.py
```

训练配置见 `config/cargo_dataset.yaml`，默认类别为 `pickup_zone` 和 `delivery_zone`。

## 测试与验证

默认测试范围由 `pytest.ini` 管理，排除了数据集、权重、日志和 vendored DroneKit 目录。

```bash
python -m pytest -q
```

常用的快速检查包括：

```bash
python -m py_compile main.py core/*.py utils/*.py scripts/*.py
python -m pytest tests/test_main.py tests/test_state_machine.py tests/test_streamer.py -q
```

硬件相关测试应在确认桨叶安全、飞控模式、供电、遥控/急停链路和实验场地后单独执行，不应混入默认单元测试。


## 文档索引

补充设计与调试说明位于 `docs/`：

- `docs/project_summary.md`: 当前项目架构摘要；
- `docs/fsm_state.md`: 状态机行为与接口契约；
- `docs/飞控串口配置.md`、`docs/数传串口配置.md`: 飞控与串口链路配置；
- `docs/首次飞行调试完整流程.md`: 实机飞行调试流程；
- `docs/无遥控器安全飞行方案.md`、`docs/无遥控器快速参考.md`: 安全飞行相关说明。

