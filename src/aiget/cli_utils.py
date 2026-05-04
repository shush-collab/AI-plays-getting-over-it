#!/usr/bin/env python3
from __future__ import annotations

import argparse

from .frame_capture import CaptureRegion


def add_capture_region_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capture-left", type=int, default=None, help="Game capture x offset.")
    parser.add_argument("--capture-top", type=int, default=None, help="Game capture y offset.")
    parser.add_argument("--capture-width", type=int, default=None, help="Game capture width.")
    parser.add_argument("--capture-height", type=int, default=None, help="Game capture height.")


def capture_region_from_args(args: argparse.Namespace) -> CaptureRegion | None:
    values = (
        args.capture_left,
        args.capture_top,
        args.capture_width,
        args.capture_height,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise SystemExit(
            "Capture region requires all of: --capture-left --capture-top "
            "--capture-width --capture-height"
        )
    assert args.capture_left is not None
    assert args.capture_top is not None
    assert args.capture_width is not None
    assert args.capture_height is not None
    return CaptureRegion(
        left=args.capture_left,
        top=args.capture_top,
        width=args.capture_width,
        height=args.capture_height,
    )
