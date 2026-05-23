# drone_Pico2 抓取/投递控制系统

## 项目简介
本项目基于 Raspberry Pi Pico 的抓取/投递控制固件，负责驱动两路舵机完成抓取与释放动作，并向飞控输出升空触发脉冲。

程序采用状态机驱动与非阻塞串口命令输入，目标是保证动作可控、按序执行，并在异常时尽量恢复到安全状态。

## 功能特性
- 单路抓取触发（带消抖）
- 双舵机 50Hz PWM 控制（角度映射到脉宽）
- 舵机安全保护机制：
  - 重复角度命令去重
  - 最小命令间隔限制
  - 过频触发冷却保护
- 串口命令与状态机绑定，避免误触发
- 错误状态自动恢复并回到空闲

## 硬件引脚定义
默认引脚定义如下（以 `drone_Pico2.c` 中的宏为准）：

- `SERVO1_PIN`: GPIO14（S1 舵机）
- `SERVO2_PIN`: GPIO15（S2 舵机）
- `GRAB_IR_PIN`: GPIO7（抓取红外，低电平触发，内部上拉）
- `FLY_TRIGGER_PIN`: GPIO16（飞控触发，高电平脉冲）
- LED: GPIO2/3/4/5（D1..D4）

约定：
- 红外被遮挡输出低电平
- 飞控触发为高电平脉冲（源码中示例为 1000 ms）

## 项目逻辑讲解

### 1. 整体控制模式
主循环采用“收命令 + 跑状态机”的轮询结构：

1. 非阻塞读取串口命令
2. 执行一次状态机推进
3. 短暂延时后继续循环

这样可以避免流程被单一步骤长期阻塞，便于调试和扩展。

### 2. 状态机流程
状态集合（源码中实际使用）：

1. `STATE_IDLE` — 空闲：等待命令
2. `STATE_GRAB` — 抓取动作（等待 IR 触发并执行 S1->S2）
3. `STATE_HOLDING` — 持货等待
4. `STATE_RELEASE` — 释放/复位动作（S1->S2）
5. `STATE_RESET` — 安全复位动作（用于错误恢复或强制复位）
6. `STATE_ERROR` — 错误态，执行安全恢复后回到空闲

关键逻辑说明：

- `STATE_IDLE`：等待 `START_GRAB` / `START_RELEASE` / `RESET` 等命令。
- `STATE_GRAB`：收到 `START_GRAB` 后，代码会等待 `GRAB_IR_PIN` 变低并通过消抖，随后在 5 秒延时后按顺序驱动 S1、再驱动 S2；动作完成后发送 `GRAB_DONE` 并进入 `STATE_HOLDING`，同时触发飞控脉冲。
- `STATE_HOLDING`：持货阶段，不主动动作，等待 `START_RELEASE`。
- `STATE_RELEASE`：执行先 S1 后 S2 的复位序列，完成后发送 `RELEASE_DONE` 并回到 `STATE_IDLE`，同时触发飞控脉冲。
- `STATE_RESET`：执行与 `RELEASE` 相似的安全复位并发送 `RESET_DONE`，用于错误恢复或手动复位。
- `STATE_ERROR`：进入错误态时尝试安全复位，随后回到 `STATE_IDLE`（受错误恢复计数限制）。

### 3. 红外判定与消抖
- 使用单一路 `GRAB_IR_PIN` 做低电平触发输入，代码实现了简单的时间窗口消抖逻辑（`IR_DEBOUNCE_MS`）。

### 4. 舵机控制与保护
- 舵机使用 50Hz PWM（`SERVO_PWM_FREQ`），角度通过 `angle_to_pwm()` 映射到脉宽。
- 保护策略包括：重复指令去重、最小命令间隔、窗口计数的过频冷却。

## 串口命令协议
命令以关键字检测（非阻塞读取），当前支持（以源码为准）：

- `START_GRAB`
- `START_RELEASE`
- `RESET`

反馈消息（源码会重复发送以提高可靠性）：

- 抓取完成：`GRAB_DONE`
- 释放完成：`RELEASE_DONE`
- 复位完成：`RESET_DONE`

命令只有在对应状态或允许的条件下被接受，不匹配时源码会输出拒绝样式的日志。

## 环境搭建（Windows）

### 1. 软件与工具
建议通过 Raspberry Pi Pico VS Code 扩展安装与管理工具链。

当前工程配置参考：
- Pico SDK: 2.2.0
- Ninja: 1.12.1
- picotool: 2.2.0-a4

### 2. VS Code 准备
安装扩展：
- Raspberry Pi Pico
- C/C++

然后在 VS Code 打开本工程根目录。

### 3. SDK 环境
首次使用 Pico 扩展时完成 SDK 与工具链初始化。项目 CMake 会自动引用用户目录下的 `.pico-sdk`。

### 4. 编译
本工程已配置 VS Code 任务，直接运行：
- `Compile Project`

或者命令行方式：

```powershell
cmake -B build -G Ninja
cmake --build build
```

### 5. 下载/烧录
可按硬件连接方式选择：
- Run Project（使用 picotool）
- Flash（使用 openocd）

## 切换 TEST_MODE（0 / 1）

`TEST_MODE` 为编译期宏（值 0 或 1）：

- `TEST_MODE=1`：测试模式，使用 USB CDC（`stdio`）作为命令与日志通道，并启用 `DBG_PRINT`。
- `TEST_MODE=0`：现场模式，命令输入使用 UART0（通常映射到 GP12/GP13），USB 仍可用于刷写/下载。

切换示例（命令行）：

```powershell
cmake -S . -B build -G Ninja -DTEST_MODE=1   # 或 -DTEST_MODE=0
cmake --build build
```

## 目录说明
- `drone_Pico2.c`：主程序（状态机、命令解析、红外与舵机控制）
- `CMakeLists.txt`：构建配置
- `pico_sdk_import.cmake`：Pico SDK 导入脚本
- `build/`：构建输出目录

## 调试建议
- 上电后先观察初始化日志
- 重点关注流程反馈：
  - `GRAB_DONE` / `RELEASE_DONE` / `RESET_DONE`
- 若频繁进入 `STATE_ERROR`：
  - 检查红外接线与逻辑
  - 检查舵机供电是否稳定且共地
  - 确认命令发送顺序满足状态机要求

## 版权与致谢
请以源码为最终依据，本 README 仅对代码中明显不一致的描述进行了修正。

