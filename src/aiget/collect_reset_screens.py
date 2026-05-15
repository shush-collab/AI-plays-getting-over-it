#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

from .cli_utils import (
    add_capture_region_args,
    capture_region_from_args,
    parse_args_allowing_launch_flags,
)
from .frame_capture import CaptureRegion, FrameCapture
from .memory_probe import auto_pid
from .png_utils import write_gray_png
from .window_control import WindowControl, WindowGeometry

LABEL_BY_KEY = {
    "c": "title_continue",
    "n": "title_new_game",
    "d": "confirm_dialog",
    "s": "settings",
    "p": "playable",
    "l": "dark_loading",
}


def _default_launch_command() -> list[str]:
    return [
        "steam",
        "-applaunch",
        "240720",
        "--",
        "-screen-fullscreen",
        "0",
        "-screen-width",
        "1280",
        "-screen-height",
        "720",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect full-size reset/menu screenshots with labels and metadata."
    )
    parser.add_argument("--output-dir", default="runs/reset_screens")
    parser.add_argument("--launch-command", nargs="+", default=_default_launch_command())
    parser.add_argument("--clean-save-path", default="~/goi_reset_saves/start_clean")
    parser.add_argument(
        "--active-save-path",
        default="~/.config/unity3d/Bennett Foddy/Getting Over It",
    )
    parser.add_argument("--window-title", default="Getting Over It")
    parser.add_argument("--window-left", type=int, default=None)
    parser.add_argument("--window-top", type=int, default=None)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--no-kill-existing", action="store_true")
    parser.add_argument("--no-launch", action="store_true", help="Attach to already-running game.")
    add_capture_region_args(parser)
    args = parse_args_allowing_launch_flags(parser)

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_launch:
        if not args.no_kill_existing:
            _kill_existing_game()
        _restore_save(args.clean_save_path, args.active_save_path)
        subprocess.Popen(args.launch_command)

    pid = _wait_for_game_process(timeout=60.0)
    control = _wait_for_window(args.window_title, timeout=60.0)
    control.move_resize(
        left=args.window_left,
        top=args.window_top,
        width=args.window_width,
        height=args.window_height,
    )
    geometry = control.geometry()
    capture_region = capture_region_from_args(args) or _region_from_geometry(geometry)

    print(f"Saving frames to {run_dir}")
    print(
        "Press labels while it records: "
        "c=continue n=new-game d=dialog s=settings p=playable l=loading"
    )
    _collect_frames(run_dir, pid, geometry, capture_region, args.seconds, args.interval)


def _collect_frames(
    run_dir: Path,
    pid: int,
    geometry: WindowGeometry,
    capture_region: CaptureRegion,
    seconds: float,
    interval: float,
) -> None:
    capture = FrameCapture(
        output_shape=(capture_region.height, capture_region.width, 1),
        region=capture_region,
        allow_blank=False,
        prefer_xwd=True,
    )
    reader = _KeyReader()
    last_label = "unlabeled"
    started = time.monotonic()
    index = 0
    try:
        with reader:
            next_frame = time.monotonic()
            while time.monotonic() - started < seconds:
                key = reader.read_key()
                if key in LABEL_BY_KEY:
                    last_label = LABEL_BY_KEY[key]
                    print(f"label={last_label}")

                now = time.monotonic()
                if now < next_frame:
                    time.sleep(min(0.05, next_frame - now))
                    continue
                frame = capture.read_full_gray()
                stem = f"frame_{index:03d}"
                path = run_dir / f"{stem}.png"
                write_gray_png(path, frame)
                meta = {
                    "pid": pid,
                    "window_id": geometry.window_id,
                    "window_left": geometry.x,
                    "window_top": geometry.y,
                    "window_width": geometry.width,
                    "window_height": geometry.height,
                    "capture_left": capture_region.left,
                    "capture_top": capture_region.top,
                    "capture_width": capture_region.width,
                    "capture_height": capture_region.height,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "label": last_label,
                    "timestamp": time.time(),
                }
                (run_dir / f"{stem}.meta.json").write_text(
                    json.dumps(meta, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(
                    f"{stem}: {last_label} "
                    f"mean={float(frame.mean()):.3f} std={float(frame.std()):.3f}"
                )
                index += 1
                next_frame += interval
    finally:
        capture.close()


class _KeyReader:
    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old_attrs = None

    def __enter__(self):
        if sys.stdin.isatty():
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def read_key(self) -> str | None:
        if not sys.stdin.isatty():
            return None
        readable, _writable, _error = select.select([sys.stdin], [], [], 0)
        if not readable:
            return None
        return sys.stdin.read(1).lower()


def _wait_for_window(title: str, *, timeout: float) -> WindowControl:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        control = WindowControl(title=title)
        try:
            control.focus()
            return control
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5)
    raise RuntimeError(f"WINDOW_NOT_FOUND: {last_exc}") from last_exc


def _wait_for_game_process(*, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            pid = auto_pid()
        except RuntimeError:
            time.sleep(0.5)
            continue
        if _game_process_ready(pid):
            return pid
        time.sleep(0.5)
    raise RuntimeError("PROCESS_TIMEOUT: GettingOverIt.x86_64 did not become ready")


def _game_process_ready(pid: int) -> bool:
    try:
        maps = Path(f"/proc/{pid}/maps").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "GameAssembly.so" in maps and "UnityPlayer.so" in maps


def _kill_existing_game() -> None:
    while True:
        try:
            pid = auto_pid()
        except RuntimeError:
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(pid, signal.SIGKILL)


def _restore_save(clean_save_path: str, active_save_path: str) -> None:
    clean = Path(clean_save_path).expanduser()
    active = Path(active_save_path).expanduser()
    if not clean.exists():
        raise RuntimeError(f"clean save path does not exist: {clean}")
    active.parent.mkdir(parents=True, exist_ok=True)
    if clean.is_dir():
        if active.exists() and not active.is_dir():
            active.unlink()
        if active.exists():
            shutil.rmtree(active)
        shutil.copytree(clean, active)
    else:
        if active.exists() and active.is_dir():
            shutil.rmtree(active)
        shutil.copy2(clean, active)


def _region_from_geometry(geometry: WindowGeometry) -> CaptureRegion:
    return CaptureRegion(
        left=geometry.x,
        top=geometry.y,
        width=geometry.width,
        height=geometry.height,
    )


if __name__ == "__main__":
    main()
