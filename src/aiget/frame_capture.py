#!/usr/bin/env python3
from __future__ import annotations

import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


class FrameCapture:
    def __init__(
        self,
        *,
        output_shape: tuple[int, int, int] = (84, 84, 1),
        region: CaptureRegion | None = None,
        allow_blank: bool = True,
        xwd_window_class: str = "GettingOverIt",
        prefer_xwd: bool = False,
    ):
        self.output_shape = output_shape
        self.region = region
        self.allow_blank = allow_blank
        self.xwd_window_class = xwd_window_class
        self.prefer_xwd = prefer_xwd
        self._blank = np.zeros(output_shape, dtype=np.uint8)
        self._sct = None
        self._xwd_window_id: str | None = None

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def read(self, out: np.ndarray | None = None) -> np.ndarray:
        target = self._blank.copy() if out is None else out
        try:
            frame = self._capture_raw()
        except Exception:
            if not self.allow_blank:
                raise
            target.fill(0)
            return target

        gray = _bgra_to_gray(frame)
        resized = _resize_nearest(gray, self.output_shape[0], self.output_shape[1])
        target[:, :, 0] = resized
        return target

    def read_full_gray(self) -> np.ndarray:
        frame = self._capture_raw()
        return _bgra_to_gray(frame)

    def read_full_bgra(self) -> np.ndarray:
        return self._capture_raw().copy()

    def _capture_raw(self) -> np.ndarray:
        if self.prefer_xwd:
            return self._capture_xwd_window()
        if self._sct is None:
            import mss

            self._sct = mss.mss()
        if self.region is None:
            monitor = self._sct.monitors[1]
        else:
            monitor = {
                "left": self.region.left,
                "top": self.region.top,
                "width": self.region.width,
                "height": self.region.height,
            }
        try:
            frame = np.asarray(self._sct.grab(monitor), dtype=np.uint8)
        except Exception:
            if self.region is None:
                raise
            return self._capture_xwd_window()
        if self.region is not None and frame[:, :, :3].max(initial=0) == 0:
            return self._capture_xwd_window()
        return frame

    def _capture_xwd_window(self) -> np.ndarray:
        last_exc: Exception | None = None
        window_ids = [self._xwd_window_id] if self._xwd_window_id is not None else []
        window_ids.append(None)
        for window_id in window_ids:
            try:
                resolved_window_id = window_id or _find_x_window(self.xwd_window_class)
                frame = _decode_xwd(_run_xwd(resolved_window_id))
            except Exception as exc:
                last_exc = exc
                self._xwd_window_id = None
                continue
            self._xwd_window_id = resolved_window_id
            return frame
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("xwd capture failed")


def _find_x_window(window_class: str) -> str:
    completed = subprocess.run(
        ["xdotool", "search", "--onlyvisible", "--class", window_class],
        check=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    ids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not ids:
        raise RuntimeError(f"no visible X window found for class {window_class!r}")
    return ids[-1]


def _run_xwd(window_id: str) -> bytes:
    with tempfile.NamedTemporaryFile(prefix="aiget-xwd-", suffix=".xwd", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            ["xwd", "-silent", "-id", window_id, "-out", str(tmp_path)],
            check=True,
            capture_output=True,
            timeout=2.0,
        )
        data = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
    if not data:
        raise RuntimeError(f"xwd returned no data for window {window_id}")
    return data


def _decode_xwd(data: bytes) -> np.ndarray:
    if len(data) < 100:
        raise ValueError("XWD data too short")

    header = _unpack_xwd_header(data[:100])
    (
        header_size,
        _file_version,
        _pixmap_format,
        _pixmap_depth,
        width,
        height,
        _xoffset,
        _byte_order,
        _bitmap_unit,
        _bitmap_bit_order,
        _bitmap_pad,
        bits_per_pixel,
        bytes_per_line,
        _visual_class,
        red_mask,
        green_mask,
        blue_mask,
        _bits_per_rgb,
        _colormap_entries,
        ncolors,
        _window_width,
        _window_height,
        _window_x,
        _window_y,
        _window_bdrwidth,
    ) = header

    if bits_per_pixel != 32:
        raise ValueError(f"unsupported XWD bits_per_pixel={bits_per_pixel}")
    if (red_mask, green_mask, blue_mask) != (0xFF0000, 0x00FF00, 0x0000FF):
        raise ValueError(
            "unsupported XWD channel masks: "
            f"red={red_mask:#x} green={green_mask:#x} blue={blue_mask:#x}"
        )
    if width <= 0 or height <= 0 or bytes_per_line < width * 4:
        raise ValueError("invalid XWD dimensions")

    payload_offset = header_size + ncolors * 12
    payload_size = bytes_per_line * height
    if len(data) < payload_offset + payload_size:
        raise ValueError("truncated XWD pixel payload")

    raw = data[payload_offset : payload_offset + payload_size]
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_line)
    return rows[:, : width * 4].reshape(height, width, 4)


def _unpack_xwd_header(data: bytes) -> tuple[int, ...]:
    for endian in (">", "<"):
        header = struct.unpack(endian + "25I", data)
        header_size, file_version, _format, _depth, width, height = header[:6]
        if file_version == 7 and 100 <= header_size <= 4096 and width > 0 and height > 0:
            return header
    raise ValueError("invalid XWD header")


def _bgra_to_gray(frame: np.ndarray) -> np.ndarray:
    blue = frame[:, :, 0].astype(np.uint16)
    green = frame[:, :, 1].astype(np.uint16)
    red = frame[:, :, 2].astype(np.uint16)
    return ((77 * red + 150 * green + 29 * blue) >> 8).astype(np.uint8)


def _resize_nearest(gray: np.ndarray, height: int, width: int) -> np.ndarray:
    y_idx = np.linspace(0, gray.shape[0] - 1, height).astype(np.intp)
    x_idx = np.linspace(0, gray.shape[1] - 1, width).astype(np.intp)
    return gray[y_idx][:, x_idx]
