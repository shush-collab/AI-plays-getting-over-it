#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .png_utils import gray_to_rgb, read_png, to_gray, write_rgb_png

FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    ",": ("000", "000", "000", "010", "100"),
    "-": ("000", "000", "111", "000", "000"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay coordinate grid on reset screenshot.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    image = gray_to_rgb(to_gray(read_png(args.image)))
    annotated = draw_grid(image, step=args.step)
    output = args.output or args.image.with_name(f"{args.image.stem}_grid.png")
    write_rgb_png(output, annotated)
    print(output)


def draw_grid(image: np.ndarray, *, step: int = 50) -> np.ndarray:
    out = image.copy()
    height, width, _channels = out.shape
    for x in range(0, width, step):
        out[:, x : x + 1] = (255, 0, 0)
        _draw_text(out, x + 2, 2, str(x), color=(255, 255, 0))
    for y in range(0, height, step):
        out[y : y + 1, :] = (255, 0, 0)
        _draw_text(out, 2, y + 2, str(y), color=(255, 255, 0))
    return out


def _draw_text(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    *,
    color: tuple[int, int, int],
) -> None:
    cursor = x
    for char in text:
        glyph = FONT.get(char)
        if glyph is None:
            cursor += 4
            continue
        _draw_glyph(image, cursor, y, glyph, color=color)
        cursor += 4


def _draw_glyph(
    image: np.ndarray,
    x: int,
    y: int,
    glyph: tuple[str, ...],
    *,
    color: tuple[int, int, int],
) -> None:
    height, width, _channels = image.shape
    for row, bits in enumerate(glyph):
        py = y + row
        if py < 0 or py >= height:
            continue
        for col, bit in enumerate(bits):
            px = x + col
            if bit == "1" and 0 <= px < width:
                image[py, px] = color


if __name__ == "__main__":
    main()
