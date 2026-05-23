#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/uart.h"
#include "hardware/pwm.h"
#include "hardware/clocks.h"

#ifndef TEST_MODE
#define TEST_MODE 0
#endif

// Debug print macro: when TEST_MODE is defined (debug mode), enable DBG_PRINT to
// print verbose logs; in field mode DBG_PRINT is a no-op.
#if TEST_MODE
#define DBG_PRINT(...) printf(__VA_ARGS__)
#else
#define DBG_PRINT(...) do {} while(0)
#endif

// ===================== 硬件配置（按实际接线修改）=====================
#define SERVO1_PIN 14       // S1舵机引脚
#define SERVO2_PIN 15       // S2舵机引脚
#define FLY_TRIGGER_PIN 16 // 飞控升空触发引脚
#define LED_D1 2       // LED（红）
#define GRAB_IR_PIN 7      // 抓取红外触发输入（低电平触发）
#define LED_D2 3       // LED（黄）
#define LED_D3 4       // LED（蓝）
#define LED_D4 5       // LED（绿）


#define IR_DEBOUNCE_MS 40  // 红外输入消抖时间
// ===================== 舵机参数配置 =====================
// 常见舵机使用 50Hz PWM，0~180 度通常映射到 0.5ms~2.5ms 高电平脉宽。
#define SERVO_PWM_FREQ 50       // 标准50Hz

typedef struct {
    bool initialized;
    bool last_raw_low;
    bool debounced_low;
    absolute_time_t last_change_time;
} DebounceInput;

static DebounceInput grab_ir_debounce = {0};
#define SERVO_PWM_WRAP 19999    // 20ms 对应 20000 计数（wrap+1）
#define SERVO_TARGET_ANGLE 90   // 取货时旋转角度
#define SERVO_RESET_ANGLE  0    // 投递时复位角度
#define SERVO_DELAY_MS 2000     // 抓取动作等待时间
#define SERVO_GRAB_GAP_MS 1000   // 取货时两舵机动作间隔
#define SERVO_MIN_CMD_INTERVAL_MS 250   // 同一舵机两次命令最小间隔
#define SERVO_OVERFREQ_WINDOW_MS 1200   // 过频统计窗口
#define SERVO_OVERFREQ_LIMIT 5          // 窗口内最大允许动作次数
#define SERVO_PROTECT_COOLDOWN_MS 2000  // 触发保护后的冷却时间

// ===================== 状态机定义 =====================
// 最小命令驱动状态机：收到 grab 就抓取，收到 release 就释放。
typedef enum {
    STATE_IDLE,                 // 空闲：等待指令
    STATE_GRAB,                 // 抓取动作：S1->S2
    STATE_HOLDING,              // 持货等待
    STATE_RELEASE,              // 释放动作：S1->S2
    STATE_RESET,                // 安全复位动作：S1->S2
    STATE_ERROR
} SystemState;

// ===================== 全局变量 =====================
// 串口缓冲只做短命令接收（START_GRAB / START_RELEASE / RESET）。
static SystemState current_state = STATE_IDLE;
static char uart_buf[64] = {0};
static uint8_t buf_idx = 0;
// 错误恢复计数器 - 避免ERROR状态无限循环
static uint8_t error_recovery_count = 0;
static const uint8_t ERROR_RECOVERY_MAX = 1;  // ERROR状态最多执行1次恢复
int DBG1 = 0;
int DBG2 = 0;

// ===================== 舵机保护结构体 =====================
// 无电流传感器条件下的软件防堵转：
// 1) 过滤重复角度命令，避免无意义顶死输出；
// 2) 限制最小动作间隔，避免短时间频繁反复；
// 3) 过频后进入冷却，降低持续堵转风险。
typedef struct {
    uint8_t commanded_angle;
    absolute_time_t last_cmd_time;
    absolute_time_t window_start_time;
    absolute_time_t protect_until;
    uint8_t window_cmd_count;
    bool cooldown_notified;
    bool initialized;
    // 状态恢复机制 - 记录该舵机的失败重试状态
    uint8_t fail_retry_count;
} ServoGuard;

typedef enum {
    SERVO_ACTION_DONE,
    SERVO_ACTION_SKIPPED,
    SERVO_ACTION_BLOCKED
} ServoActionResult;

static ServoGuard servo1_guard = {0};
static ServoGuard servo2_guard = {0};
static absolute_time_t actuate_start = {0};  // 使用 {0} 初始化为nil_time
static absolute_time_t state_enter_time = {0};
static SystemState last_state_observed = (SystemState)-1;
static bool fly_pulse_active = false;
static absolute_time_t fly_pulse_until = {0};
static bool grab_seq_started = false;
static absolute_time_t grab_seq_gap_start = {0};
static bool grab_ir_wait_started = false;
static absolute_time_t grab_ir_trigger_time = {0};
static bool release_seq_started = false;
static absolute_time_t release_seq_gap_start = {0};
static bool reset_seq_started = false;
static absolute_time_t reset_seq_gap_start = {0};
static bool reset_done = false; // whether RESET command has been received

typedef struct {
    const char *keyword;
    SystemState required_state;
    bool allow_any_state;
    SystemState next_state;
    const char *received_msg;
    const char *rejected_msg;
} CommandRule;

// forward declarations for servo helpers (used by command dispatch)
ServoActionResult servo1_set(uint8_t angle);
ServoActionResult servo2_set(uint8_t angle);
// forward declaration for feedback helper
static void send_feedback_repeat(const char *msg, int times, int interval_ms);

static int read_command_char_nonblocking(void) {
#if TEST_MODE
    // Test mode: keep command source on USB stdio.
    return getchar_timeout_us(0);
#else
    // Field mode: command input is forced to UART0 only.
    if (uart_is_readable(uart0)) {
        return uart_getc(uart0);
    }
    return PICO_ERROR_TIMEOUT;
#endif
}

static bool uart_try_dispatch_command(void) {
    static const CommandRule rules[] = {
        {"START_GRAB",    STATE_IDLE,     false, STATE_GRAB,    "[CMD] Received: START_GRAB\r\n",    "[CMD] Rejected: START_GRAB (invalid state %d)\r\n"},
        {"START_RELEASE", STATE_IDLE,     true,  STATE_RELEASE, "[CMD] Received: START_RELEASE\r\n", "[CMD] Rejected: START_RELEASE (invalid state %d)\r\n"},
        {"RESET",         STATE_IDLE,     true,  STATE_RESET,   "[CMD] Received: RESET\r\n",         "[CMD] Rejected: RESET (invalid state %d)\r\n"},
    };

    for (uint i = 0; i < sizeof(rules) / sizeof(rules[0]); i++) {
            if (strstr(uart_buf, rules[i].keyword) != NULL) {
            if (current_state == rules[i].required_state ||
                rules[i].allow_any_state) {
                DBG_PRINT("%s", rules[i].received_msg);
                current_state = rules[i].next_state;
            } else {
                DBG_PRINT(rules[i].rejected_msg, current_state);
            }
            buf_idx = 0;
            memset(uart_buf, 0, sizeof(uart_buf));
            return true;
        }
    }

    return false;
}

// ===================== LED 灯效系统 =====================
// 若你的LED是低电平点亮，请改为 1。
#define LED_ACTIVE_LOW 1
#define LED_FX_TICK_MS 50

typedef struct {
    absolute_time_t last_tick;
    uint32_t tick;
    uint32_t lfsr;
    SystemState last_state;
} LedFxState;

static LedFxState led_fx = {0};

static inline void led_write_pin(uint pin, bool on) {
    gpio_put(pin, LED_ACTIVE_LOW ? !on : on);
}

static void led_apply_mask(uint8_t mask) {
    led_write_pin(LED_D1, (mask & 0x01u) != 0);
    led_write_pin(LED_D2, (mask & 0x02u) != 0);
    led_write_pin(LED_D3, (mask & 0x04u) != 0);
    led_write_pin(LED_D4, (mask & 0x08u) != 0);
}

static uint32_t led_rand32(void) {
    uint32_t x = led_fx.lfsr;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    led_fx.lfsr = x;
    return x;
}

void led_fx_init(void) {
    gpio_init(LED_D1); gpio_set_dir(LED_D1, GPIO_OUT);
    gpio_init(LED_D2); gpio_set_dir(LED_D2, GPIO_OUT);
    gpio_init(LED_D3); gpio_set_dir(LED_D3, GPIO_OUT);
    gpio_init(LED_D4); gpio_set_dir(LED_D4, GPIO_OUT);

    led_fx.last_tick = get_absolute_time();
    led_fx.tick = 0;
    led_fx.lfsr = 0xA5A55A5Au;
    led_fx.last_state = current_state;

    // 开机灯序：点亮扫描 + 全亮收束
    const uint8_t boot_seq[] = {0x1, 0x2, 0x4, 0x8, 0x4, 0x2, 0x1, 0xF, 0x0};
    for (uint i = 0; i < sizeof(boot_seq); i++) {
        led_apply_mask(boot_seq[i]);
        sleep_ms(55);
    }
}

void led_fx_update(void) {
    absolute_time_t now = get_absolute_time();
    if (absolute_time_diff_us(led_fx.last_tick, now) < (int64_t)LED_FX_TICK_MS * 1000) {
        return;
    }
    led_fx.last_tick = now;

    if (led_fx.last_state != current_state) {
        led_fx.tick = 0;
        led_fx.last_state = current_state;
    }

    uint8_t mask = 0;
    switch (current_state) {
        case STATE_IDLE: {
            if (!reset_done) {
                // 未收到 RESET，常亮红灯
                mask = 0x1;
            } else {
                // 只亮绿、蓝灯，表示系统准备就绪
                const uint8_t seq[] = {0x8, 0x8, 0x8,0x8,0x8,0xC,0xC,0xC,0xC,0xC};
                mask = seq[led_fx.tick % (sizeof(seq) / sizeof(seq[0]))];
            }
            break;
        }
        case STATE_GRAB:
        case STATE_RELEASE: {
            // 彗星推进：逐步拉满再熄灭
            const uint8_t seq[] = {0x1, 0x3, 0x7, 0xF, 0xE, 0xC, 0x8, 0x0};
            mask = seq[led_fx.tick % (sizeof(seq) / sizeof(seq[0]))];
            break;
        }
        case STATE_RESET: {
            // 复位过程沿用同样的推进灯效，表示正在执行安全回位。
            const uint8_t seq[] = {0x1, 0x3, 0x7, 0xF, 0xE, 0xC, 0x8, 0x0};
            mask = seq[led_fx.tick % (sizeof(seq) / sizeof(seq[0]))];
            break;
        }
        case STATE_HOLDING: {
            // 持货巡航：骑士流光：左右来回扫描
            const uint8_t seq[] = {0x1, 0x2, 0x4, 0x8, 0x4, 0x2};
            mask = seq[led_fx.tick % (sizeof(seq) / sizeof(seq[0]))];
            break;
        }
        case STATE_ERROR: {
            // 错误态：爆闪 + 随机火花
            if ((led_fx.tick & 1u) == 0u) {
                mask = 0xFu;
            } else {
                uint32_t r = led_rand32();
                mask = (uint8_t)(r & 0xFu);
                if (mask == 0) {
                    mask = (uint8_t)(1u << ((r >> 4) & 0x3u));
                }
            }
            break;
        }
        default:
            mask = 0;
            break;
    }

    led_apply_mask(mask);
    led_fx.tick++;
}

static inline int64_t state_elapsed_ms(void) {
    if (is_nil_time(state_enter_time)) {
        return 0;
    }
    return absolute_time_diff_us(state_enter_time, get_absolute_time()) / 1000;
}

static bool grab_ir_is_triggered(void) {
    absolute_time_t now = get_absolute_time();
    bool raw_low = (gpio_get(GRAB_IR_PIN) == 0);

    if (!grab_ir_debounce.initialized) {
        grab_ir_debounce.initialized = true;
        grab_ir_debounce.last_raw_low = raw_low;
        // 首次采样不直接判定为稳定态，确保至少经历一次消抖窗口。
        grab_ir_debounce.debounced_low = false;
        grab_ir_debounce.last_change_time = now;
        return false;
    }

    if (raw_low != grab_ir_debounce.last_raw_low) {
        grab_ir_debounce.last_raw_low = raw_low;
        grab_ir_debounce.last_change_time = now;
    }

    if (raw_low != grab_ir_debounce.debounced_low) {
        if (absolute_time_diff_us(grab_ir_debounce.last_change_time, now)
            >= (int64_t)IR_DEBOUNCE_MS * 1000) {
            grab_ir_debounce.debounced_low = raw_low;
        }
    }

    return grab_ir_debounce.debounced_low;
}

void fly_trigger_pulse_start(uint32_t pulse_ms) {
    gpio_put(FLY_TRIGGER_PIN, 1);
    fly_pulse_active = true;
    fly_pulse_until = delayed_by_ms(get_absolute_time(), pulse_ms);
}

void fly_trigger_update(void) {
    if (fly_pulse_active && absolute_time_diff_us(get_absolute_time(), fly_pulse_until) <= 0) {
        gpio_put(FLY_TRIGGER_PIN, 0);
        fly_pulse_active = false;
    }
}

// ===================== 舵机驱动函数 =====================
// 角度转 PWM 计数值：
// - PWM 时基配置为 1MHz（1 计数=1us）
// - 周期 20ms（wrap=19999）
// - 角度线性映射到 500us~2500us
uint16_t angle_to_pwm(uint8_t angle) {
    if (angle > 180) angle = 180;
    // 使用浮点数除法避免整数截断导致角度精度丢失
    float pulse_us = 500.0f + ((float)angle / 180.0f) * 2000.0f;
    float period_us = 1000000.0f / (float)SERVO_PWM_FREQ;
    uint32_t level = (uint32_t)((pulse_us / period_us) * (float)(SERVO_PWM_WRAP + 1u));
    if (level > SERVO_PWM_WRAP) level = SERVO_PWM_WRAP;
    return (uint16_t)level;
}

// ===================== 通用的安全驱动内核 =====================
// 返回值：
// - DONE: 已执行动作
// - SKIPPED: 目标角度与当前角度一致，属于正常空操作
// - BLOCKED: 被保护逻辑拦截
ServoActionResult servo_safe_set_internal(ServoGuard *guard, uint pin, uint8_t target_angle) {
    absolute_time_t now = get_absolute_time();

    // 1. 懒初始化
    if (!guard->initialized) {
        guard->commanded_angle = SERVO_RESET_ANGLE;
        guard->last_cmd_time = nil_time;
        guard->window_start_time = now;
        guard->protect_until = now;
        guard->window_cmd_count = 0;
        guard->cooldown_notified = false;
        guard->initialized = true;
    }

    // 2. 冷却期检查
    // absolute_time_diff_us(from, to) = to - from
    // 若 now < protect_until，则 (protect_until - now) > 0，表示仍在冷却期内
    if (absolute_time_diff_us(now, guard->protect_until) > 0) {
        if (!guard->cooldown_notified) {
            DBG_PRINT("[WARN] Servo locked. Cooldown remaining: %lld ms\r\n", 
                   absolute_time_diff_us(now, guard->protect_until) / 1000);
            guard->cooldown_notified = true;
        }
        DBG_PRINT("[WARN] Command to pin %d blocked due to cooldown.\r\n", pin);
        return SERVO_ACTION_BLOCKED;
    }
    guard->cooldown_notified = false;

    // 3. 相同指令去重
    if (target_angle == guard->commanded_angle) {
        return SERVO_ACTION_SKIPPED;
    }

    // 4. 最小间隔限制 (如果太快，直接拒绝，不做 sleep 阻塞)
    if (!is_nil_time(guard->last_cmd_time)) {
        int64_t dt_us = absolute_time_diff_us(guard->last_cmd_time, now);
        if (dt_us < (int64_t)SERVO_MIN_CMD_INTERVAL_MS * 1000) {
            DBG_PRINT("[WARN] Command to pin %d blocked due to minimum interval. Time since last command: %lld ms\r\n", 
                   pin, dt_us / 1000);
            return SERVO_ACTION_BLOCKED;
        }
    }

    // 5. 滑动窗口过频保护
    int64_t win_us = absolute_time_diff_us(guard->window_start_time, now);
    if (win_us > (int64_t)SERVO_OVERFREQ_WINDOW_MS * 1000) {
        guard->window_start_time = now;
        guard->window_cmd_count = 0;
    }
    
    guard->window_cmd_count++;
    if (guard->window_cmd_count > SERVO_OVERFREQ_LIMIT) {
        guard->protect_until = delayed_by_ms(now, SERVO_PROTECT_COOLDOWN_MS);
        DBG_PRINT("[ERROR] Servo overheated! Locking for %d ms.\r\n", SERVO_PROTECT_COOLDOWN_MS);
        return SERVO_ACTION_BLOCKED;
    }

    // 6. 执行动作 (这里只发令，不 sleep!)
    pwm_set_gpio_level(pin, angle_to_pwm(target_angle));
    
    // 7. 更新状态
    guard->commanded_angle = target_angle;
    guard->last_cmd_time = now;
    
    DBG_PRINT("[SERVO pin %d] Set to %d deg\r\n", pin, target_angle);
    // 移除阻塞sleep - servo_safe_set_internal应该是非阻塞的
    // 舵机的实际运行延迟由调用端(状态机)处理
    return SERVO_ACTION_DONE;
}

// 舵机初始化：两路舵机统一配置为 50Hz，并默认回到复位角。
void servo_init(void) {
    uint slice1 = pwm_gpio_to_slice_num(SERVO1_PIN);
    uint slice2 = pwm_gpio_to_slice_num(SERVO2_PIN);

    // 按当前系统时钟计算分频，避免不同默认时钟(如 125MHz/150MHz)导致频率偏差。
    uint32_t clk_hz = clock_get_hz(clk_sys);
    float clkdiv = (float)clk_hz / ((float)(SERVO_PWM_WRAP + 1u) * (float)SERVO_PWM_FREQ);
    if (clkdiv < 1.0f) clkdiv = 1.0f;
    if (clkdiv > 255.0f) clkdiv = 255.0f;

    pwm_set_clkdiv(slice1, clkdiv);
    pwm_set_wrap(slice1, SERVO_PWM_WRAP);
    pwm_set_clkdiv(slice2, clkdiv);
    pwm_set_wrap(slice2, SERVO_PWM_WRAP);

    float actual_freq = (float)clk_hz / (clkdiv * (float)(SERVO_PWM_WRAP + 1u));
        DBG_PRINT("[PWM] clk_sys=%lu Hz, clkdiv=%.3f, wrap=%u, target=%d Hz, actual=%.3f Hz\r\n",
            (unsigned long)clk_hz, clkdiv, SERVO_PWM_WRAP, SERVO_PWM_FREQ, actual_freq);

    // 上电先让 S1 单独进入复位，再延时后接上 S2，避免两路同时动作。
    gpio_set_function(SERVO1_PIN, GPIO_FUNC_PWM);
    pwm_set_gpio_level(SERVO1_PIN, angle_to_pwm(SERVO_RESET_ANGLE));

    gpio_set_function(SERVO2_PIN, GPIO_FUNC_PWM);
    pwm_set_gpio_level(SERVO2_PIN, angle_to_pwm(SERVO_RESET_ANGLE));
    pwm_set_enabled(slice1, true);
    pwm_set_enabled(slice2, true);
}

// 独立控制 S1，动作后阻塞等待舵机到位。
ServoActionResult servo1_set(uint8_t angle) {
    return servo_safe_set_internal(&servo1_guard, SERVO1_PIN, angle);
}
// 独立控制 S2，动作后阻塞等待舵机到位。
ServoActionResult servo2_set(uint8_t angle) {
    return servo_safe_set_internal(&servo2_guard, SERVO2_PIN, angle);
}

// ===================== 串口指令解析 =====================
// 采用非阻塞读取：缓冲区里一旦出现关键字就立即触发，不再依赖换行符。
void uart_process_command(void) {
    int c = read_command_char_nonblocking();
    while (c != PICO_ERROR_TIMEOUT) {
        if (c == '\n' || c == '\r') {
            uart_buf[buf_idx] = '\0';
            buf_idx = 0;
            memset(uart_buf, 0, sizeof(uart_buf));
        } else if (buf_idx < sizeof(uart_buf)-1) {
            uart_buf[buf_idx++] = c;
            uart_buf[buf_idx] = '\0';

            if (uart_try_dispatch_command()) {
                c = read_command_char_nonblocking();
                continue;
            }
        } else {
            // 溢出保护
                DBG_PRINT("[WARN] UART buffer overflow\r\n");
            buf_idx = 0;
            memset(uart_buf, 0, sizeof(uart_buf));
        }
        c = read_command_char_nonblocking();
    }
}

// ===================== 主状态机 =====================
// 规则摘要：
// 1) IDLE 收到 grab 后，等待 GPIO7 红外低电平触发再执行抓取；
// 2) HOLDING 收到 release 直接执行释放动作；
// 3) 抓取触发输入带消抖处理。
void state_machine_run(void) {
    bool entered = false;
    if (current_state != last_state_observed) {
        last_state_observed = current_state;
        state_enter_time = get_absolute_time();
        entered = true;

        if (current_state != STATE_GRAB) {
            grab_seq_started = false;
            grab_seq_gap_start = nil_time;
            grab_ir_wait_started = false;
            grab_ir_trigger_time = nil_time;
        }
        if (current_state != STATE_RELEASE) {
            release_seq_started = false;
            release_seq_gap_start = nil_time;
        }
        if (current_state != STATE_RESET) {
            reset_seq_started = false;
            reset_seq_gap_start = nil_time;
        }
    }

    switch (current_state) {
        case STATE_IDLE:
            if (entered) {
                DBG_PRINT("[STATE] IDLE (Send 'START_GRAB')\r\n");
            }
            break;

        case STATE_GRAB: {
            if (!grab_seq_started) {
                if (!grab_ir_is_triggered()) {
                    if (entered) {
                        DBG_PRINT("[WAIT] START_GRAB received, waiting IR(GPIO7) low trigger...\r\n");
                    }
                    grab_ir_wait_started = false;
                    grab_ir_trigger_time = nil_time;
                    break;
                }

                if (!grab_ir_wait_started) {
                    grab_ir_wait_started = true;
                    grab_ir_trigger_time = get_absolute_time();
                    DBG_PRINT("[WAIT] IR triggered, delaying 5 seconds before grab...\r\n");
                    break;
                }

                if (absolute_time_diff_us(grab_ir_trigger_time, get_absolute_time())
                    < 5000 * 1000) {
                    break;
                }

                DBG_PRINT("[ACT] 5-second delay elapsed, drive S1 90deg, then S2 90deg\r\n");
                ServoActionResult res1 = servo1_set(SERVO_TARGET_ANGLE+15);
                if (res1 == SERVO_ACTION_BLOCKED) {
                    DBG_PRINT("[WARN] S1 blocked, will retry\r\n");
                    break;
                }
                grab_seq_started = true;
                grab_seq_gap_start = get_absolute_time();
                break;
            }

            if (absolute_time_diff_us(grab_seq_gap_start, get_absolute_time())
                < (int64_t)SERVO_GRAB_GAP_MS * 1000) {
                break;
            }

            ServoActionResult res2 = servo2_set(SERVO_TARGET_ANGLE+3);
            if (res2 == SERVO_ACTION_BLOCKED) {
                DBG_PRINT("[WARN] S2 blocked, will retry\r\n");
                break;
            }
            if (is_nil_time(actuate_start)) {
                actuate_start = get_absolute_time();
            }
            
            if (absolute_time_diff_us(actuate_start, get_absolute_time()) 
                > SERVO_DELAY_MS * 1000) {
                // 舵机动作完成后给上位机反馈，再等待升空指令。
                DBG_PRINT("[FEEDBACK] grab_finished\r\n");
                send_feedback_repeat("GRAB_DONE\r\n", 3, 50);
                // grab 动作结束后立即通知飞控，避免把升空信号延后到命令阶段。
                fly_trigger_pulse_start(1000);
                actuate_start = nil_time;
                current_state = STATE_HOLDING;
                grab_seq_started = false;
                grab_seq_gap_start = nil_time;
                grab_ir_wait_started = false;
                grab_ir_trigger_time = nil_time;
            }
            break;
        }

        case STATE_HOLDING:
            // 持货等待阶段，不主动动作，仅等待 release 命令。
            if (entered) {
                DBG_PRINT("[STATE] Holding (Send 'START_RELEASE')\r\n");
            }
            break;

        case STATE_RELEASE: {
            if (!release_seq_started) {
                DBG_PRINT("[ACT] Release sequence: S1 then S2\r\n");
                // 按既定机械顺序先 S1 后 S2，避免机构干涉。
                ServoActionResult s1_res = servo1_set(SERVO_RESET_ANGLE);
                if (s1_res == SERVO_ACTION_BLOCKED) {
                    DBG_PRINT("[WARN] S1 reset blocked, will retry\r\n");
                    break;
                }
                release_seq_started = true;
                release_seq_gap_start = get_absolute_time();
                break;
            }

            if (absolute_time_diff_us(release_seq_gap_start, get_absolute_time())
                < (int64_t)SERVO_GRAB_GAP_MS * 1000) {
                break;
            }

            ServoActionResult s2_res = servo2_set(SERVO_RESET_ANGLE);
            if (s2_res == SERVO_ACTION_BLOCKED) {
                DBG_PRINT("[WARN] S2 reset blocked, will retry\r\n");
                break;
            }

            if (is_nil_time(actuate_start)) {
                actuate_start = get_absolute_time();
            }

            if (absolute_time_diff_us(actuate_start, get_absolute_time())
                > SERVO_DELAY_MS * 1000) {
                DBG_PRINT("[FEEDBACK] release_finished\r\n");
                send_feedback_repeat("RELEASE_DONE\r\n", 3, 50);
                // release 动作结束后立即通知飞控。
                fly_trigger_pulse_start(1000);
                actuate_start = nil_time;
                current_state = STATE_IDLE;
                release_seq_started = false;
                release_seq_gap_start = nil_time;
            }
            break;
        }

        case STATE_RESET: {
            if (!reset_seq_started) {
                DBG_PRINT("[ACT] Reset sequence: S1 then S2\r\n");
                gpio_put(FLY_TRIGGER_PIN, 0);
                fly_pulse_active = false;
                actuate_start = nil_time;
                error_recovery_count = 0;
                reset_done = false;

                ServoActionResult s1_res = servo1_set(SERVO_RESET_ANGLE);
                if (s1_res == SERVO_ACTION_BLOCKED) {
                    DBG_PRINT("[WARN] S1 reset blocked, will retry\r\n");
                    break;
                }
                reset_seq_started = true;
                reset_seq_gap_start = get_absolute_time();
                break;
            }

            if (absolute_time_diff_us(reset_seq_gap_start, get_absolute_time())
                < (int64_t)SERVO_GRAB_GAP_MS * 1000) {
                break;
            }

            ServoActionResult s2_res = servo2_set(SERVO_RESET_ANGLE);
            if (s2_res == SERVO_ACTION_BLOCKED) {
                DBG_PRINT("[WARN] S2 reset blocked, will retry\r\n");
                break;
            }

        
            if (is_nil_time(actuate_start)) {
                actuate_start = get_absolute_time();
            }

            if (absolute_time_diff_us(actuate_start, get_absolute_time())
                > SERVO_DELAY_MS * 1000) {
                DBG_PRINT("[FEEDBACK] reset_finished\r\n");
                send_feedback_repeat("RESET_DONE\r\n", 3, 50);
                reset_done = true;
                actuate_start = nil_time;
                current_state = STATE_IDLE;
                reset_seq_started = false;
                reset_seq_gap_start = nil_time;
            }
            break;
        }

        case STATE_ERROR:
            // 错误恢复：执行安全复位并回到空闲态。
            if (entered) {
                DBG_PRINT("[STATE] ERROR. Resetting to IDLE...\r\n");
                servo1_set(SERVO_RESET_ANGLE);
                servo2_set(SERVO_RESET_ANGLE);
                gpio_put(FLY_TRIGGER_PIN, 0);
                fly_pulse_active = false;
                actuate_start = nil_time;
                error_recovery_count++;
            }

            if (error_recovery_count >= ERROR_RECOVERY_MAX && state_elapsed_ms() >= 2000) {
                DBG_PRINT("[STATE] ERROR recovery complete. Back to IDLE.\r\n");
                error_recovery_count = 0;
                current_state = STATE_IDLE;
            }
            break;

        default:
            current_state = STATE_IDLE;
            break;
    }
}

// ===================== 主函数 =====================
// 主循环采用“收命令 + 跑状态机”的轮询模式。
int main() {
    stdio_init_all();
    sleep_ms(2000);
    #if TEST_MODE
    DBG_PRINT("[MODE] TEST_MODE: USB stdio\r\n");
    #else
    DBG_PRINT("[MODE] FIELD_MODE: UART0 on GP12/GP13\r\n");
    #endif
    DBG_PRINT("\r\n=== drone_Pico2 Grab/Release System Ready ===\r\n");

    // 初始化飞控触发引脚，默认低电平。
    gpio_init(FLY_TRIGGER_PIN);
    gpio_set_dir(FLY_TRIGGER_PIN, GPIO_OUT);
    gpio_put(FLY_TRIGGER_PIN, 0);

    // 初始化抓取红外输入：GPIO7，低电平触发，开启上拉防止悬空。
    gpio_init(GRAB_IR_PIN);
    gpio_set_dir(GRAB_IR_PIN, GPIO_IN);
    gpio_pull_up(GRAB_IR_PIN);

    servo_init();
    led_fx_init();
    DBG_PRINT("Init done.\r\n");

    while (1) {
        uart_process_command();
        fly_trigger_update();
        state_machine_run();
        led_fx_update();
        sleep_ms(20);
    }
}

// 将反馈信息重复发送若干次，增加上位机/串口丢包容错。
static void send_feedback_repeat(const char *msg, int times, int interval_ms) {
    if (times <= 0) return;
    for (int i = 0; i < times; ++i) {
        printf("%s", msg);
        // 让传输有短暂时间窗口，USB CDC/串口驱动会在此期间发送数据
        sleep_ms(interval_ms);
    }
}