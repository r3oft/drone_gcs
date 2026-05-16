# FSM 状态机与接口契约

最后校准日期：2026-05-16

本文档描述当前 `core/state_machine.py` 的实际行为。早期 Phase 2 协作契约、`MavlinkCommander`、MCU 经 Pixhawk UART4 透传等设计已被当前实现替代。

## 核心依赖

`GlobalFSM` 构造参数：

- `IFlightBridge`：飞控连接、遥测、起飞、速度控制、降落、模式切换。
- `IMCUBridge`：MCU 通信，默认 PC 直连串口。
- `TargetPoseEstimator`：YOLO-OBB 目标位姿提取。
- `VisualServoController`：PD 视觉伺服速度输出。
- `ConfigManager`：运行参数。

当前真实实现：

| 接口 | 默认实现 | 说明 |
|---|---|---|
| `IFlightBridge` | `core.flight_bridge.FlightBridge` | Pixhawk MAVLink 链路 |
| `IMCUBridge` | `core.mcu_bridge.DirectSerialMCUBridge` | PC 直连 MCU 串口 |
| `IMCUBridge` legacy | `core.flight_bridge.MCUBridge` | Pixhawk `SERIAL_CONTROL` 兼容路径 |
| mock | `utils.mock` | 桌面无硬件联调 |

## 状态列表

| 状态 | 含义 |
|---|---|
| `IDLE` | 空闲，等待 `request_start()` |
| `RESET` | 连接飞控和 MCU，发送 `RESET`，等待 `RESET_DONE` 与飞控心跳 |
| `INBOUND` | 起飞到 `flight.takeoff_alt` |
| `TASK_REC_ALIGN` | 对准 pickup zone，目标类别 `cls_id=0` |
| `TASK_REC_DESCEND` | pickup 区下降，保持平面纠偏 |
| `TASK_REC_WAIT_LOAD` | 发送 `START_GRAB`，等待 `GRAB_DONE` |
| `TRANS_DELIVERY` | 二次起飞并定距飞往 delivery 区 |
| `TASK_REL_ALIGN` | 对准 delivery zone，目标类别 `cls_id=1` |
| `TASK_REL_DESCEND` | delivery 区下降 |
| `TASK_REL_RELEASE` | 发送 `START_RELEASE`，等待 `RELEASE_DONE` |
| `OUTBOUND` | 调用 `land()`，落地后回到 `IDLE` |
| `EMERGENCY` | 请求 RTL，由飞控 failsafe 接管 |

## 状态迁移

```text
IDLE
  -> RESET
  -> INBOUND
  -> TASK_REC_ALIGN
  -> TASK_REC_DESCEND
  -> TASK_REC_WAIT_LOAD
  -> TRANS_DELIVERY
  -> TASK_REL_ALIGN
  -> TASK_REL_DESCEND
  -> TASK_REL_RELEASE
  -> OUTBOUND
  -> IDLE
```

异常分支：

- 飞控断连、心跳丢失、任务中被外部切到 RTL/LAND：进入 `EMERGENCY`。
- `arm_and_takeoff()` 或 `land()` 返回失败：进入 `EMERGENCY`。
- MCU 命令发送失败、失败响应/超时重试耗尽：进入 `EMERGENCY`。

## RESET

`RESET` 首次 tick 执行：

1. `flight.connect()`
2. `mcu.connect()`
3. `mcu.send_command(MCUCommand.RESET)`

随后每 tick 检查：

- `flight.get_telemetry()["heartbeat_ok"]`
- `mcu.get_latest_response() == MCUResponse.RESET_DONE`

两者都满足后进入 `INBOUND`。`RESET` 超过 `_RESET_TIMEOUT_S` 进入 `EMERGENCY`。

## 视觉阶段

视觉阶段由 `_VISION_CLS_MAP` 绑定目标类别：

```python
TASK_REC_ALIGN   -> 0
TASK_REC_DESCEND -> 0
TASK_REL_ALIGN   -> 1
TASK_REL_DESCEND -> 1
```

`_run_vision_pipeline()` 负责：

1. 调用 `perception.process_frame(frame, cls_id)`。
2. 目标存在时更新 `_last_target_seen`。
3. 调用 `controller.compute_velocity(target, center_u, center_v, dt)`。
4. 返回 `(vx, vy, vyaw)`。

目标丢失策略：

- 丢失时间小于 `fsm.target_lost_hover_s`：不额外发新速度。
- 超过 `fsm.target_lost_hover_s`：发送零速度悬停。
- 超过 `fsm.target_lost_climb_s`：发送 `fsm.climb_vz` 爬升搜索。

进入视觉状态时会重置 `_last_target_seen`，防止首帧无目标立即爬升。

## 对齐与下降

对齐阶段：

- 平面速度命令：`flight.send_body_velocity(vx, vy, 0.0, vyaw)`
- 当 `(vx, vy, vyaw)` 连续为零达到 `fsm.align_hold_time_s` 后进入下降。

下降阶段：

- 先检查飞控遥测：`alt < flight.land_detect_alt` 或 `armed == False` 视为触地/落地。
- 未触地时继续视觉纠偏。
- 平面速度乘以 `_DESCEND_GAIN=0.5`，垂直速度使用 `_DESCEND_VZ=0.15`。

## MCU 动作

MCU 动作统一由 `_handle_mcu_action()` 处理：

| 阶段 | 命令 | 成功响应 | 失败响应 | 成功后 |
|---|---|---|---|---|
| `TASK_REC_WAIT_LOAD` | `START_GRAB` | `GRAB_DONE` | `GRAB_FAIL` | `TRANS_DELIVERY` |
| `TASK_REL_RELEASE` | `START_RELEASE` | `RELEASE_DONE` | `RELEASE_FAIL` | `OUTBOUND` |

超时参数：

- `mcu.grab_timeout_s`
- `mcu.release_timeout_s`
- `mcu.retry_max`

默认 MCU 协议见 `飞控串口配置.md`。

## 运行入口

mock：

```bash
python main.py --mode mock --start --mock-fast --duration 2 --no-flight-recorder
```

live：

```bash
python main.py --mode live --start --confirm-live-start
```

live 自动启动必须显式传入 `--confirm-live-start`，避免误触发实机任务。

## 测试

默认单元测试：

```bash
python -m pytest -q
```

重点覆盖：

- `tests/test_state_machine.py`
- `tests/test_main.py`
- `tests/test_mcu_bridge.py`
- `tests/test_mock.py`
