#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import os
import struct
import time
from dataclasses import dataclass
from typing import Protocol

EV_SYN = 0x00
EV_REL = 0x02
REL_X = 0x00
REL_Y = 0x01
SYN_REPORT = 0x00
BUS_USB = 0x03

UI_SET_EVBIT = 0x40045564
UI_SET_RELBIT = 0x40045566
UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502


class ActionSender(Protocol):
    def send_mouse_delta(self, action: tuple[float, float]) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass
class NullActionSender:
    max_dx: int = 40
    max_dy: int = 40

    def send_mouse_delta(self, action: tuple[float, float]) -> None:
        return None

    def close(self) -> None:
        return None


class UInputMouseSender:
    def __init__(self, device: str = "/dev/uinput", *, max_dx: int = 40, max_dy: int = 40):
        self.device = device
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.fd = os.open(device, os.O_WRONLY | os.O_NONBLOCK)
        self._create_device()

    def _create_device(self) -> None:
        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_REL)
        fcntl.ioctl(self.fd, UI_SET_RELBIT, REL_X)
        fcntl.ioctl(self.fd, UI_SET_RELBIT, REL_Y)

        name = b"aiget-rl-mouse"
        payload = struct.pack(
            "80sHHHHi" + "i" * 256,
            name,
            BUS_USB,
            1,
            1,
            1,
            0,
            *([0] * 256),
        )
        os.write(self.fd, payload)
        fcntl.ioctl(self.fd, UI_DEV_CREATE)
        time.sleep(0.1)

    def send_mouse_delta(self, action: tuple[float, float]) -> None:
        dx = int(max(-1.0, min(1.0, float(action[0]))) * self.max_dx)
        dy = int(max(-1.0, min(1.0, float(action[1]))) * self.max_dy)
        if dx:
            self._emit(EV_REL, REL_X, dx)
        if dy:
            self._emit(EV_REL, REL_Y, dy)
        self._emit(EV_SYN, SYN_REPORT, 0)

    def close(self) -> None:
        if self.fd < 0:
            return
        try:
            fcntl.ioctl(self.fd, UI_DEV_DESTROY)
        finally:
            os.close(self.fd)
            self.fd = -1

    def _emit(self, event_type: int, code: int, value: int) -> None:
        os.write(self.fd, struct.pack("llHHi", 0, 0, event_type, code, int(value)))


def open_action_sender(
    *,
    enabled: bool = True,
    device: str = "/dev/uinput",
    max_dx: int = 40,
    max_dy: int = 40,
) -> ActionSender:
    if not enabled:
        return NullActionSender(max_dx=max_dx, max_dy=max_dy)
    return UInputMouseSender(device=device, max_dx=max_dx, max_dy=max_dy)
