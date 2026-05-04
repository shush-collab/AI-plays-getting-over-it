#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import time
import zlib
from pathlib import Path

import numpy as np

from .cli_utils import add_capture_region_args, capture_region_from_args
from .env import IMAGE_OBS_KEY, RESET_RELAUNCH, GettingOverItEnv


def _default_launch_command() -> list[str]:
    return ["steam", "-applaunch", "240720"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify relaunch/save-restore reset for Getting Over It."
    )
    parser.add_argument("--resets", type=int, default=5, help="Number of reset attempts.")
    parser.add_argument(
        "--launch-command",
        nargs="+",
        default=_default_launch_command(),
        help="Command used to launch the game.",
    )
    parser.add_argument(
        "--clean-save-path",
        default="~/goi_reset_saves/start_clean",
        help="Known clean save/checkpoint path.",
    )
    parser.add_argument(
        "--active-save-path",
        default="~/.config/unity3d/Bennett Foddy/Getting Over It",
        help="Runtime save path restored before each reset.",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Base env frame rate.")
    parser.add_argument("--image-hz", type=float, default=30.0, help="Image capture rate.")
    parser.add_argument("--image-std-threshold", type=float, default=10.0)
    parser.add_argument("--image-mean-min", type=float, default=5.0)
    parser.add_argument("--image-mean-max", type=float, default=250.0)
    parser.add_argument("--sleep-after-reset", type=float, default=3.0)
    parser.add_argument("--game-ready-timeout", type=float, default=45.0)
    parser.add_argument("--startup-click", nargs=2, type=int, metavar=("X", "Y"))
    parser.add_argument("--startup-key", type=str, default=None)
    parser.add_argument("--startup-delay", type=float, default=0.0)
    parser.add_argument("--startup-attempts", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="runs/reset_test",
        help="Directory for reset frame PNGs.",
    )
    parser.add_argument(
        "--no-kill-existing",
        action="store_true",
        help="Do not terminate any existing game process before relaunch.",
    )
    add_capture_region_args(parser)
    args = parser.parse_args()

    capture_region = capture_region_from_args(args)
    if capture_region is None:
        raise SystemExit("Reset test requires explicit game capture region.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = GettingOverItEnv(
        reset_backend=RESET_RELAUNCH,
        launch_command=args.launch_command,
        clean_save_path=args.clean_save_path,
        active_save_path=args.active_save_path,
        kill_existing_on_reset=not args.no_kill_existing,
        game_ready_timeout=args.game_ready_timeout,
        enable_uinput=False,
        strict_image=True,
        image_std_threshold=args.image_std_threshold,
        image_mean_min=args.image_mean_min,
        image_mean_max=args.image_mean_max,
        startup_click=tuple(args.startup_click) if args.startup_click is not None else None,
        startup_key=args.startup_key,
        startup_delay=args.startup_delay,
        startup_attempts=args.startup_attempts,
        capture_region=capture_region,
        dt=1.0 / args.hz,
        image_hz=args.image_hz,
    )
    try:
        for index in range(args.resets):
            try:
                obs, info = env.reset()
            except Exception as exc:
                trace = env.reset_trace
                frame_path = output_dir / f"reset_{index}_failure.png"
                _write_last_frame_if_available(env, frame_path)
                print(f"RESET {index} FAILED")
                print(f"  failure_stage: {trace.get('failure_stage', '')}")
                print(f"  reason: {trace.get('reason') or exc}")
                print(f"  old_pid: {trace.get('old_pid', '')}")
                print(f"  new_pid: {trace.get('new_pid', '')}")
                print(f"  process_ready_ms: {trace.get('process_ready_ms', '')}")
                print(f"  modules_ready_ms: {trace.get('modules_ready_ms', '')}")
                print(f"  image_ready_ms: {trace.get('image_ready_ms', '')}")
                print(f"  startup_action_sent: {trace.get('startup_action_sent', False)}")
                print(f"  playercontrol_ready_ms: {trace.get('playercontrol_ready_ms', '')}")
                print(f"  fast_cursor_addr: {trace.get('fast_cursor_addr', '')}")
                print(f"  frame: {frame_path if frame_path.exists() else ''}")
                env.cleanup_reset_processes()
                raise
            image = obs[IMAGE_OBS_KEY]
            frame_path = output_dir / f"reset_{index}.png"
            _write_gray_png(frame_path, image[:, :, -1])
            trace = info["reset_trace"]

            print(f"RESET {index}")
            print(f"  old_pid: {trace.get('old_pid', '')}")
            print(f"  pid: {info['pid']}")
            print(f"  reset_mode: {info['reset_mode']}")
            print(f"  process_ready_ms: {trace.get('process_ready_ms', '')}")
            print(f"  modules_ready_ms: {trace.get('modules_ready_ms', '')}")
            print(f"  image_ready_ms: {trace.get('image_ready_ms', '')}")
            print(f"  startup_action_sent: {trace.get('startup_action_sent', False)}")
            print(f"  playercontrol_ready_ms: {trace.get('playercontrol_ready_ms', '')}")
            print(f"  fast_cursor_addr: {trace.get('fast_cursor_addr', '')}")
            print(f"  image_mean: {float(info['image_mean']):.6f}")
            print(f"  image_std: {float(info['image_std']):.6f}")
            print(f"  image_age: {float(info['image_age']):.6f}")
            print(f"  image_updates: {info['image_updates']}")
            print(f"  process_lost: {info['process_lost']}")
            print(f"  frame: {frame_path}")

            if info["reset_mode"] != "relaunch_save_restore":
                raise RuntimeError(f"unexpected reset_mode: {info['reset_mode']}")
            if bool(info["process_lost"]):
                raise RuntimeError("process_lost after reset")
            if float(info["image_std"]) <= args.image_std_threshold:
                raise RuntimeError(
                    "image_std too low after reset: "
                    f"{float(info['image_std']):.6f} <= {args.image_std_threshold:.6f}"
                )
            if not (args.image_mean_min <= float(info["image_mean"]) <= args.image_mean_max):
                raise RuntimeError(
                    "image_mean out of playable bounds after reset: "
                    f"{float(info['image_mean']):.6f} not in "
                    f"[{args.image_mean_min:.6f}, {args.image_mean_max:.6f}]"
                )
            time.sleep(args.sleep_after_reset)
    finally:
        env.close()


def _write_gray_png(path: Path, image: np.ndarray) -> None:
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


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _write_last_frame_if_available(env: GettingOverItEnv, path: Path) -> None:
    try:
        image = env._image_latest[:, :, -1]  # noqa: SLF001
    except Exception:
        return
    if image.size:
        _write_gray_png(path, image)


if __name__ == "__main__":
    main()
