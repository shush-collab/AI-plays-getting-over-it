#!/usr/bin/env python3
from __future__ import annotations

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
    def __init__(self, *, title: str = "Getting Over It", window_class: str = "GettingOverIt"):
        self.title = title
        self.window_class = window_class
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
        window_id = self.focus()
        _run(
            ["xdotool", "mousemove", "--window", window_id, str(x), str(y)],
            timeout=3.0,
        )
        _run(["xdotool", "click", "1"], timeout=3.0)

    def key(self, key: str) -> None:
        window_id = self.focus()
        _run(["xdotool", "key", "--window", window_id, key], timeout=3.0)


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
