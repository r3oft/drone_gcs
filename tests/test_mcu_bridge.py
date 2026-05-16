from core.interfaces import MCUCommand, MCUResponse
from core.mcu_bridge import DirectSerialMCUBridge, MCUConfig


class FakeSerial:
    def __init__(self, read_lines=None, fail_write=False):
        self.read_lines = list(read_lines or [])
        self.fail_write = fail_write
        self.is_open = True
        self.writes = []
        self.flushed = False
        self.reset_input_called = False

    def reset_input_buffer(self):
        self.reset_input_called = True

    def write(self, data):
        if self.fail_write:
            raise OSError("write failed")
        self.writes.append(data)
        return len(data)

    def flush(self):
        self.flushed = True

    def readline(self):
        if self.read_lines:
            return self.read_lines.pop(0)
        return b""

    def close(self):
        self.is_open = False


def make_bridge(monkeypatch, fake_serial=None, factory_error=None):
    fake = fake_serial or FakeSerial()
    calls = []

    def fake_serial_for_url(*args, **kwargs):
        calls.append((args, kwargs))
        if factory_error is not None:
            raise factory_error
        return fake

    monkeypatch.setattr("core.mcu_bridge.serial.serial_for_url", fake_serial_for_url)
    bridge = DirectSerialMCUBridge(
        MCUConfig(
            port="loop://",
            baudrate=57600,
            read_timeout_s=0.01,
            write_timeout_s=0.2,
        )
    )
    return bridge, fake, calls


def test_direct_serial_mcu_connect_opens_serial(monkeypatch):
    bridge, fake, calls = make_bridge(monkeypatch)

    assert bridge.connect() is True
    assert bridge.is_connected()
    assert fake.reset_input_called is True
    assert calls[0][0] == ("loop://",)
    assert calls[0][1]["baudrate"] == 57600
    assert calls[0][1]["timeout"] == 0.01
    assert calls[0][1]["write_timeout"] == 0.2


def test_direct_serial_mcu_send_command_writes_line(monkeypatch):
    bridge, fake, _ = make_bridge(monkeypatch)

    assert bridge.send_command(MCUCommand.START_GRAB) is True

    assert fake.writes == [b"START_GRAB\n"]
    assert fake.flushed is True


def test_direct_serial_mcu_reads_known_responses(monkeypatch):
    bridge, _, _ = make_bridge(
        monkeypatch,
        FakeSerial([b"GRAB_DONE\n", b"RELEASE_DONE\r\n"]),
    )
    assert bridge.connect()

    assert bridge.get_latest_response() == MCUResponse.GRAB_DONE
    assert bridge.get_latest_response() == MCUResponse.RELEASE_DONE


def test_direct_serial_mcu_waits_for_complete_line(monkeypatch):
    bridge, _, _ = make_bridge(monkeypatch, FakeSerial([b"GRAB_", b"DONE\n"]))
    assert bridge.connect()

    assert bridge.get_latest_response() is None
    assert bridge.get_latest_response() == MCUResponse.GRAB_DONE


def test_direct_serial_mcu_ignores_unknown_response(monkeypatch):
    bridge, _, _ = make_bridge(monkeypatch, FakeSerial([b"NOISE\n"]))
    assert bridge.connect()

    assert bridge.get_latest_response() is None


def test_direct_serial_mcu_send_returns_false_when_connect_fails(monkeypatch):
    bridge, _, _ = make_bridge(monkeypatch, factory_error=OSError("no port"))

    assert bridge.send_command(MCUCommand.START_GRAB) is False


def test_direct_serial_mcu_send_returns_false_when_write_fails(monkeypatch):
    bridge, _, _ = make_bridge(monkeypatch, FakeSerial(fail_write=True))

    assert bridge.send_command(MCUCommand.START_RELEASE) is False


def test_direct_serial_mcu_rejects_invalid_command(monkeypatch):
    bridge, fake, _ = make_bridge(monkeypatch)

    assert bridge.send_command("BAD_CMD") is False
    assert fake.writes == []
