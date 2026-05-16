from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import serial

from core.interfaces import IMCUBridge, MCUCommand, MCUResponse


logger = logging.getLogger("MCUBridge")


@dataclass
class MCUConfig:
    """Configuration for a direct PC serial link to the MCU."""

    port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    read_timeout_s: float = 0.02
    write_timeout_s: float = 0.5
    encoding: str = "ascii"


class DirectSerialMCUBridge(IMCUBridge):
    """Line-oriented MCU bridge over a direct serial port."""

    def __init__(self, config: MCUConfig | None = None) -> None:
        self.config = config or MCUConfig()
        self._serial: Any | None = None
        self._rx_buffer = bytearray()

    def connect(self) -> bool:
        if self.is_connected():
            return True

        try:
            self._serial = serial.serial_for_url(
                self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.read_timeout_s,
                write_timeout=self.config.write_timeout_s,
            )
            self._rx_buffer.clear()
            reset_input = getattr(self._serial, "reset_input_buffer", None)
            if callable(reset_input):
                reset_input()
            logger.info(
                "MCU serial connected: %s @ %s",
                self.config.port,
                self.config.baudrate,
            )
            return True
        except Exception as exc:
            logger.error("MCU serial connect failed: %s", exc, exc_info=True)
            self._serial = None
            return False

    def send_command(self, command: str) -> bool:
        if command not in MCUCommand.ALL:
            logger.error("Invalid MCU command: %s", command)
            return False

        if not self.is_connected() and not self.connect():
            logger.error("Cannot send MCU command while serial link is closed")
            return False

        assert self._serial is not None
        payload = f"{command}\n".encode(self.config.encoding)
        try:
            self._serial.write(payload)
            flush = getattr(self._serial, "flush", None)
            if callable(flush):
                flush()
            logger.info("Sent MCU command: %s", command)
            return True
        except Exception as exc:
            logger.error("Failed to send MCU command %s: %s", command, exc, exc_info=True)
            return False

    def get_latest_response(self) -> str | None:
        if not self.is_connected():
            return None

        assert self._serial is not None
        try:
            line = self._serial.readline()
        except Exception as exc:
            logger.error("Failed to read MCU response: %s", exc, exc_info=True)
            return None

        if not line:
            return None
        self._rx_buffer.extend(line)
        if b"\n" not in self._rx_buffer:
            return None

        raw_response, remaining = self._rx_buffer.split(b"\n", 1)
        self._rx_buffer = bytearray(remaining)
        response = raw_response.decode(self.config.encoding, errors="ignore").strip()
        if not response:
            return None
        if response not in MCUResponse.ALL:
            logger.warning("Ignoring unknown MCU response: %s", response)
            return None

        logger.info("Received MCU response: %s", response)
        return response

    def is_connected(self) -> bool:
        if self._serial is None:
            return False
        return bool(getattr(self._serial, "is_open", True))

    def close(self) -> None:
        if self._serial is None:
            return
        try:
            self._serial.close()
        except Exception as exc:
            logger.warning("Failed to close MCU serial cleanly: %s", exc)
        finally:
            self._serial = None
