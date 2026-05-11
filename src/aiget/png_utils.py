#!/usr/bin/env python3
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_gray_png(path: Path, image: np.ndarray) -> None:
    gray = np.asarray(image, dtype=np.uint8)
    if gray.ndim != 2:
        raise ValueError(f"expected 2D grayscale image, got shape {gray.shape}")
    height, width = gray.shape
    raw_rows = b"".join(b"\x00" + gray[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw_rows))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected RGB image, got shape {rgb.shape}")
    height, width, _channels = rgb.shape
    raw_rows = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw_rows))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def read_png(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG: {path}")
    pos = 8
    width = height = None
    idat = bytearray()
    color_type = None
    bit_depth = None
    while pos < len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        kind = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width = int.from_bytes(payload[0:4], "big")
            height = int.from_bytes(payload[4:8], "big")
            bit_depth = payload[8]
            color_type = payload[9]
        elif kind == b"IDAT":
            idat.extend(payload)
        elif kind == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or color_type not in (0, 2):
        raise ValueError(f"unsupported PNG: {path}")
    channels = 1 if color_type == 0 else 3
    raw = zlib.decompress(bytes(idat))
    stride = width * channels + 1
    rows = []
    previous = np.zeros(width * channels, dtype=np.uint8)
    for row in range(height):
        start = row * stride
        filter_type = raw[start]
        encoded = np.frombuffer(raw[start + 1 : start + stride], dtype=np.uint8).copy()
        decoded = _decode_filter(filter_type, encoded, previous, channels)
        rows.append(decoded)
        previous = decoded
    image = np.vstack(rows)
    if channels == 1:
        return image.reshape(height, width)
    return image.reshape(height, width, channels)


def to_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        red = array[:, :, 0].astype(np.uint16)
        green = array[:, :, 1].astype(np.uint16)
        blue = array[:, :, 2].astype(np.uint16)
        return ((77 * red + 150 * green + 29 * blue) >> 8).astype(np.uint8)
    raise ValueError(f"expected grayscale or RGB image, got shape {array.shape}")


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError(f"expected 2D grayscale image, got shape {image.shape}")
    return np.repeat(image[:, :, None], 3, axis=2)


def _decode_filter(
    filter_type: int,
    row: np.ndarray,
    previous: np.ndarray,
    channels: int,
) -> np.ndarray:
    if filter_type == 0:
        return row
    out = row.astype(np.uint16)
    prev = previous.astype(np.uint16)
    if filter_type == 1:
        for i in range(channels, len(out)):
            out[i] = (out[i] + out[i - channels]) & 0xFF
    elif filter_type == 2:
        out = (out + prev) & 0xFF
    elif filter_type == 3:
        for i in range(len(out)):
            left = out[i - channels] if i >= channels else 0
            up = prev[i]
            out[i] = (out[i] + ((left + up) // 2)) & 0xFF
    elif filter_type == 4:
        for i in range(len(out)):
            left = out[i - channels] if i >= channels else 0
            up = prev[i]
            up_left = prev[i - channels] if i >= channels else 0
            out[i] = (out[i] + _paeth(left, up, up_left)) & 0xFF
    else:
        raise ValueError(f"unsupported PNG filter type {filter_type}")
    return out.astype(np.uint8)


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    pa = abs(estimate - left)
    pb = abs(estimate - up)
    pc = abs(estimate - up_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return up_left


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)
