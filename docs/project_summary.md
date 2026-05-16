# 项目当前架构总览

最后校准日期：2026-05-16

本项目是基于视觉识别的微型物流无人机地面站控制软件。当前代码重点服务于室内固定场景下的 pickup zone / delivery zone 识别、视觉伺服对准、飞控速度指令下发，以及 MCU 末端执行器闭环确认。

## 当前分层

1. 决策层：地面站 PC
   - 入口：`main.py`
   - 任务调度：`core/state_machine.py`
   - 视觉推理：`core/perception.py`
   - 视觉伺服：`core/servo_controller.py`
   - 调试 HUD：`scripts/run_visual_hud.py`

2. 飞行控制层：Pixhawk
   - 实现：`core/flight_bridge.py`
   - 默认链路：`mavlink.connection`，当前配置为 `udp:192.168.4.1:14550`
   - 责任：连接、心跳/遥测、解锁起飞、LAND/RTL、机体系速度指令。

3. 末端执行层：MCU
   - 默认实现：`core/mcu_bridge.py::DirectSerialMCUBridge`
   - 默认链路：PC 直连串口，`mcu.transport: direct_serial`
   - 协议：ASCII 文本行，命令和反馈均以换行结束。
   - 兼容路径：`core.flight_bridge.MCUBridge` 仍保留 Pixhawk `SERIAL_CONTROL` legacy 透传。

4. 图像采集层：ESP32-S3 CameraWebServer
   - 实现：`core/streamer.py::ZeroLatencyStreamer`
   - 优先使用 `http://<host>:81/stream`，失败后 fallback 到 `http://<host>/capture`。

## 当前主任务流

`IDLE -> RESET -> INBOUND -> TASK_REC_ALIGN -> TASK_REC_DESCEND -> TASK_REC_WAIT_LOAD -> TRANS_DELIVERY -> TASK_REL_ALIGN -> TASK_REL_DESCEND -> TASK_REL_RELEASE -> OUTBOUND -> IDLE`

关键闭环点：

- `RESET`：飞控连接成功、MCU 串口连接成功、`RESET_DONE` 到达后进入任务。
- pickup：识别 `pickup_zone`，对齐后下降，触地/低高度后发送 `START_GRAB`，等待 `GRAB_DONE`。
- delivery：识别 `delivery_zone`，对齐后下降，触地/低高度后发送 `START_RELEASE`，等待 `RELEASE_DONE`。
- 异常：飞控连接断开、心跳丢失、外部强制 RTL/LAND、MCU 失败/超时超过重试次数，进入 `EMERGENCY` 并请求 RTL。

## 当前调试入口

- 离线/桌面主循环：`python main.py --mode mock --start --mock-fast --duration 2 --no-flight-recorder`
- live 主循环：需要 `--mode live --start --confirm-live-start`
- 视觉 HUD：`python scripts/run_visual_hud.py ...`
- 默认单元测试：`python -m pytest -q`

## 与旧文档的主要差异

- 不再使用独立 `MavlinkCommander` 文件；飞控功能集中在 `FlightBridge`。
- MCU 默认不再经飞控中转，而是 PC 直连串口。
- `tests/flight_tests/` 是人工硬件飞测脚本，不属于默认 pytest 单元测试范围。
