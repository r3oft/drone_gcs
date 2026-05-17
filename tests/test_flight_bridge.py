import time
from types import SimpleNamespace

from core.flight_bridge import FlightBridge, FlightConfig


class FakeMessageFactory:
    def set_position_target_local_ned_encode(self, *args):
        return {
            "frame": args[3],
            "type_mask": args[4],
            "vx": args[8],
            "vy": args[9],
            "vz": args[10],
            "yaw_rate": args[15],
        }


class FakeVehicle:
    def __init__(self, alt=0.5, armed=True, mode="GUIDED"):
        self.mode = SimpleNamespace(name=mode)
        self.armed = armed
        self.location = SimpleNamespace(
            global_relative_frame=SimpleNamespace(alt=alt)
        )
        self.heading = 0.0
        self.battery = SimpleNamespace(level=100)
        self.message_factory = FakeMessageFactory()
        self.sent_messages = []
        self.simple_goto_called = False
        self.flushed = False

    def simple_goto(self, target):
        self.simple_goto_called = True
        raise AssertionError("DroneKit simple_goto should not be used")

    def send_mavlink(self, msg):
        self.sent_messages.append(msg)
        # Simulate the aircraft responding to NED vertical velocity.
        self.location.global_relative_frame.alt = max(
            0.0,
            self.location.global_relative_frame.alt - msg["vz"] * 0.1,
        )

    def flush(self):
        self.flushed = True


def make_bridge(vehicle):
    bridge = FlightBridge(
        FlightConfig(
            heartbeat_timeout=30,
            takeoff_timeout_s=1,
            land_timeout_s=1,
            land_detect_alt=0.1,
            goto_vertical_speed=1.0,
            goto_alt_tolerance=0.02,
            goto_command_hz=100.0,
        )
    )
    bridge._vehicle = vehicle
    bridge._last_heartbeat_time = time.time()
    return bridge


def test_simple_goto_uses_vertical_velocity_instead_of_dronekit_goto():
    vehicle = FakeVehicle(alt=0.5)
    bridge = make_bridge(vehicle)

    assert bridge.simple_goto(0.3) is True

    assert vehicle.simple_goto_called is False
    assert any(msg["vz"] > 0 for msg in vehicle.sent_messages)
    assert vehicle.sent_messages[-1]["vz"] == 0
    assert vehicle.flushed is True


def test_simple_goto_climbs_with_negative_ned_velocity():
    vehicle = FakeVehicle(alt=0.3)
    bridge = make_bridge(vehicle)

    assert bridge.simple_goto(0.5) is True

    assert any(msg["vz"] < 0 for msg in vehicle.sent_messages)
    assert vehicle.sent_messages[-1]["vz"] == 0


def test_land_waits_for_touchdown_after_mode_switch():
    vehicle = FakeVehicle(alt=0.05, armed=True, mode="GUIDED")
    bridge = make_bridge(vehicle)

    assert bridge.land() is True
    assert vehicle.mode.name == "LAND"
