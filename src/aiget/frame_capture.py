#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

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
    ):
        self.output_shape = output_shape
        self.region = region
        self.allow_blank = allow_blank
        self._blank = np.zeros(output_shape, dtype=np.uint8)
        self._sct = None

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

    def _capture_raw(self) -> np.ndarray:
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
        return np.asarray(self._sct.grab(monitor), dtype=np.uint8)


def _bgra_to_gray(frame: np.ndarray) -> np.ndarray:
    blue = frame[:, :, 0].astype(np.uint16)
    green = frame[:, :, 1].astype(np.uint16)
    red = frame[:, :, 2].astype(np.uint16)
    return ((77 * red + 150 * green + 29 * blue) >> 8).astype(np.uint8)


def _resize_nearest(gray: np.ndarray, height: int, width: int) -> np.ndarray:
    y_idx = np.linspace(0, gray.shape[0] - 1, height).astype(np.intp)
    x_idx = np.linspace(0, gray.shape[1] - 1, width).astype(np.intp)
    return gray[y_idx][:, x_idx]
