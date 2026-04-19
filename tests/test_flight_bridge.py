import time
import os
import pytest

pytest.importorskip("dronekit")

from core.flight_bridge import FlightBridge, MCUBridge, FlightConfig


def run_demo_flow(flight_bridge: FlightBridge, mcu_bridge: MCUBridge) -> None:
    # 1. 连接飞控
    if not flight_bridge.connect():
        raise RuntimeError("飞控连接失败")

    # 2. 起飞到5米高度
    if not flight_bridge.arm_and_takeoff(5.0):
        raise RuntimeError("起飞失败")

    # 3. 发送速度指令（前向1m/s，悬停，无偏航）
    flight_bridge.send_body_velocity(1.0, 0.0, 0.0, 0.0)
    time.sleep(2)

    # 4. 获取遥测数据
    telemetry = flight_bridge.get_telemetry()
    assert isinstance(telemetry, dict)

    # 5. 发送MCU抓取指令
    if mcu_bridge.send_command("START_GRAB"):
        for _ in range(10):
            resp = mcu_bridge.get_latest_response()
            if resp:
                break
            time.sleep(0.5)

    # 6. 降落
    if not flight_bridge.land():
        raise RuntimeError("降落失败")


def test_demo_flow_happy_path(monkeypatch):
    cfg = FlightConfig(
        connection_string="tcp:127.0.0.1:5760",
        takeoff_timeout_s=30,
        land_timeout_s=60,
    )

    flight_bridge = FlightBridge(cfg)
    mcu_bridge = MCUBridge(flight_bridge, cfg)

    calls = {
        "connect": 0,
        "takeoff": 0,
        "velocity": 0,
        "telemetry": 0,
        "mcu_send": 0,
        "mcu_recv": 0,
        "land": 0,
    }

    monkeypatch.setattr(time, "sleep", lambda *_: None)

    monkeypatch.setattr(
        flight_bridge,
        "connect",
        lambda: calls.__setitem__("connect", calls["connect"] + 1) or True,
    )
    monkeypatch.setattr(
        flight_bridge,
        "arm_and_takeoff",
        lambda alt: calls.__setitem__("takeoff", calls["takeoff"] + 1) or True,
    )
    monkeypatch.setattr(
        flight_bridge,
        "send_body_velocity",
        lambda vx, vy, vz, yaw: calls.__setitem__("velocity", calls["velocity"] + 1),
    )
    monkeypatch.setattr(
        flight_bridge,
        "get_telemetry",
        lambda: calls.__setitem__("telemetry", calls["telemetry"] + 1)
        or {
            "armed": True,
            "mode": "GUIDED",
            "alt": 5.0,
            "heading": 90.0,
            "battery_pct": 0.8,
            "heartbeat_ok": True,
        },
    )
    monkeypatch.setattr(
        flight_bridge,
        "land",
        lambda: calls.__setitem__("land", calls["land"] + 1) or True,
    )

    monkeypatch.setattr(
        mcu_bridge,
        "send_command",
        lambda cmd: calls.__setitem__("mcu_send", calls["mcu_send"] + 1) or True,
    )

    responses = iter([None, None, "GRAB_DONE"])
    monkeypatch.setattr(
        mcu_bridge,
        "get_latest_response",
        lambda: calls.__setitem__("mcu_recv", calls["mcu_recv"] + 1) or next(responses),
    )

    run_demo_flow(flight_bridge, mcu_bridge)

    assert calls["connect"] == 1
    assert calls["takeoff"] == 1
    assert calls["velocity"] == 1
    assert calls["telemetry"] == 1
    assert calls["mcu_send"] == 1
    assert calls["mcu_recv"] >= 1
    assert calls["land"] == 1


def test_demo_flow_connect_fail(monkeypatch):
    cfg = FlightConfig()
    flight_bridge = FlightBridge(cfg)
    mcu_bridge = MCUBridge(flight_bridge, cfg)

    monkeypatch.setattr(flight_bridge, "connect", lambda: False)

    with pytest.raises(RuntimeError, match="飞控连接失败"):
        run_demo_flow(flight_bridge, mcu_bridge)


def test_get_connection_health_disconnected():
    bridge = FlightBridge(FlightConfig(heartbeat_timeout=2))

    health = bridge.get_connection_health()

    assert health["connected"] is False
    assert health["heartbeat_limit_s"] == 4
    assert health["connection_string"]


def test_is_connected_timeout_then_reconnect(monkeypatch):
    bridge = FlightBridge(
        FlightConfig(
            heartbeat_timeout=1,
            reconnect_enabled=True,
            reconnect_max_attempts=2,
            reconnect_backoff_s=0,
        )
    )

    class DummyVehicle:
        def close(self):
            return None

    bridge._vehicle = DummyVehicle()
    bridge._last_heartbeat_time = time.time() - 10

    monkeypatch.setattr(bridge, "connect", lambda: True)

    assert bridge.is_connected() is True


@pytest.mark.skipif(
    os.getenv("RUN_HW_SMOKE", "0") != "1",
    reason="仅在显式设置 RUN_HW_SMOKE=1 时执行硬件在线冒烟测试",
)
def test_hardware_online_smoke():
    cfg = FlightConfig(
        connection_string=os.getenv("FC_CONNECTION", "udp:192.168.4.1:14550"),
        heartbeat_timeout=int(os.getenv("FC_HEARTBEAT_TIMEOUT", "5")),
        reconnect_enabled=True,
        reconnect_max_attempts=2,
        reconnect_backoff_s=1.0,
    )
    bridge = FlightBridge(cfg)

    assert bridge.connect() is True

    telemetry = bridge.get_telemetry()
    assert isinstance(telemetry, dict)
    assert "mode" in telemetry
    assert "heartbeat_ok" in telemetry

    health = bridge.get_connection_health()
    assert isinstance(health, dict)
    assert health["connected"] is True