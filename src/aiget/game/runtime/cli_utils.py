#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .frame_capture import CaptureRegion


def add_capture_region_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capture-left", type=int, default=None, help="Game capture x offset.")
    parser.add_argument("--capture-top", type=int, default=None, help="Game capture y offset.")
    parser.add_argument("--capture-width", type=int, default=None, help="Game capture width.")
    parser.add_argument("--capture-height", type=int, default=None, help="Game capture height.")


def parse_args_allowing_launch_flags(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse args while allowing Steam-style flags inside --launch-command.

    argparse stops collecting ``nargs="+"`` values when it sees tokens such as
    ``-applaunch``. This helper extracts --launch-command first and stops only
    at option names known to the parser, so Steam/Unity launch flags remain
    part of the command.
    """
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--launch-command" not in raw_args:
        return parser.parse_args(raw_args)

    known_options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option != "--launch-command"
    }
    launch_command: list[str] | None = None
    normalized: list[str] = []
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        if token != "--launch-command":
            normalized.append(token)
            index += 1
            continue

        index += 1
        values: list[str] = []
        while index < len(raw_args) and raw_args[index] not in known_options:
            values.append(raw_args[index])
            index += 1
        if not values:
            parser.error("--launch-command expected at least one argument")
        launch_command = values

    args = parser.parse_args(normalized)
    args.launch_command = launch_command
    return args


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
