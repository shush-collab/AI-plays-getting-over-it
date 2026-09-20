#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowGeometry:
    window_id: str
    x: int
    y: int
    width: int
    height: int


class WindowControl:
    def __init__(
        self,
        *,
        title: str = "Getting Over It",
        window_class: str = "GettingOverIt",
        input_backend: str = "auto",
    ):
        self.title = title
        self.window_class = window_class
        if input_backend not in ("auto", "xdotool", "ydotool"):
            raise ValueError(
                "input_backend must be 'auto', 'xdotool', or 'ydotool', "
                f"got {input_backend!r}"
            )
        self.input_backend = input_backend
        self._active_input_backend: str | None = None
        self._window_id: str | None = None

    def find_window(self) -> str:
        if self._window_id is not None:
            return self._window_id
        self._window_id = _find_window(self.title, self.window_class)
        return self._window_id

    def focus(self) -> str:
        window_id = self.find_window()
        _run(["xdotool", "windowactivate", "--sync", window_id], timeout=2.0)
        return window_id

    def move_resize(
        self,
        *,
        left: int | None,
        top: int | None,
        width: int | None,
        height: int | None,
    ) -> None:
        window_id = self.focus()
        if width is not None and height is not None:
            _run(["xdotool", "windowsize", window_id, str(width), str(height)], timeout=2.0)
        if left is not None and top is not None:
            _run(["xdotool", "windowmove", window_id, str(left), str(top)], timeout=2.0)

    def geometry(self) -> WindowGeometry:
        window_id = self.find_window()
        completed = _run(
            ["xdotool", "getwindowgeometry", "--shell", window_id],
            timeout=2.0,
        )
        values: dict[str, str] = {"WINDOW": window_id}
        for line in completed.stdout.splitlines():
            key, sep, value = line.partition("=")
            if sep:
                values[key] = value
        return WindowGeometry(
            window_id=window_id,
            x=int(values.get("X", "0")),
            y=int(values.get("Y", "0")),
            width=int(values.get("WIDTH", "0")),
            height=int(values.get("HEIGHT", "0")),
        )

    def click_relative(self, x: int, y: int) -> None:
        self.focus()
        geometry = self.geometry()
        self.click_absolute(geometry.x + x, geometry.y + y)

    def click_absolute(self, x: int, y: int) -> None:
        backend = self._resolve_input_backend()
        if backend == "xdotool":
            _run(["xdotool", "mousemove", str(x), str(y), "click", "1"], timeout=3.0)
            return
        _run(["ydotool", "mousemove", "--absolute", str(x), str(y)], timeout=3.0)
        _run(["ydotool", "click", "0xC0"], timeout=3.0)

    def key(self, key: str) -> None:
        window_id = self.focus()
        backend = self._resolve_input_backend()
        if backend == "xdotool":
            _run(["xdotool", "key", "--window", window_id, key], timeout=3.0)
            return
        key_code = _ydotool_key_code(key)
        _run(["ydotool", "key", f"{key_code}:1", f"{key_code}:0"], timeout=3.0)

    @property
    def active_input_backend(self) -> str:
        return self._active_input_backend or self._resolve_input_backend()

    def _resolve_input_backend(self) -> str:
        if self._active_input_backend is not None:
            return self._active_input_backend
        backend = self.input_backend
        if backend == "auto":
            backend = (
                "ydotool"
                if _running_wayland() and shutil.which("ydotool") is not None
                else "xdotool"
            )
        if backend == "ydotool" and shutil.which("ydotool") is None:
            raise RuntimeError(
                "ydotool is required for reliable Wayland input automation but was not found"
            )
        if backend == "xdotool" and shutil.which("xdotool") is None:
            raise RuntimeError("xdotool is required for window/input automation but was not found")
        self._active_input_backend = backend
        return backend


def _find_window(title: str, window_class: str) -> str:
    errors: list[str] = []
    for selector in (["--name", title], ["--class", window_class]):
        try:
            completed = _run(
                ["xdotool", "search", "--onlyvisible", *selector],
                timeout=2.0,
            )
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        ids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if ids:
            return ids[-1]
    raise RuntimeError("WINDOW_NOT_FOUND: " + "; ".join(errors))


def _run(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        raise RuntimeError(f"{' '.join(command)} failed: {exc}") from exc


def _running_wayland() -> bool:
    return os.environ.get("XDG_SESSION_TYPE") == "wayland" or bool(
        os.environ.get("WAYLAND_DISPLAY")
    )


def _ydotool_key_code(key: str) -> int:
    normalized = key.lower()
    codes = {
        "return": 28,
        "enter": 28,
        "space": 57,
        "escape": 1,
        "esc": 1,
    }
    if normalized not in codes:
        raise RuntimeError(
            f"ydotool key automation only supports {', '.join(sorted(codes))}; got {key!r}"
        )
    return codes[normalized]
