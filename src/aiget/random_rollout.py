#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .cli_utils import (
    add_capture_region_args,
    capture_region_from_args,
    parse_args_allowing_launch_flags,
)
from .env import IMAGE_OBS_KEY, RESET_ATTACH, RESET_RELAUNCH, STATE_OBS_KEY, GettingOverItEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a random-action Getting Over It smoke rollout."
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Target PID. Defaults to the running game.",
    )
    parser.add_argument("--seconds", type=float, default=60.0, help="Rollout duration.")
    parser.add_argument("--hz", type=float, default=30.0, help="Base env frame rate.")
    parser.add_argument("--image-hz", type=float, default=30.0, help="Image capture rate.")
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=2,
        help="Repeated frames per policy action.",
    )
    parser.add_argument(
        "--send-actions",
        action="store_true",
        help="Send random actions through uinput.",
    )
    parser.add_argument(
        "--reset-backend",
        choices=(RESET_ATTACH, RESET_RELAUNCH),
        default=RESET_ATTACH,
        help="Game reset backend.",
    )
    parser.add_argument("--launch-command", nargs="+", default=None)
    parser.add_argument("--clean-save-path", type=str, default=None)
    parser.add_argument("--active-save-path", type=str, default=None)
    parser.add_argument("--game-ready-timeout", type=float, default=45.0)
    parser.add_argument("--startup-mode", choices=("none", "auto"), default="none")
    parser.add_argument("--title-click", nargs=2, type=int, default=(1275, 305))
    parser.add_argument("--confirm-click", nargs=2, type=int, default=(1275, 305))
    parser.add_argument("--window-left", type=int, default=None)
    parser.add_argument("--window-top", type=int, default=None)
    parser.add_argument("--window-width", type=int, default=None)
    parser.add_argument("--window-height", type=int, default=None)
    parser.add_argument(
        "--strict-image",
        action="store_true",
        help="Fail if image capture is blank, stale, or unavailable.",
    )
    parser.add_argument(
        "--discover-rich-layout",
        action="store_true",
        help="Resolve rich memory fields so reward progress debug can use body/progress masks.",
    )
    parser.add_argument(
        "--memory-window",
        type=lambda value: int(value, 0),
        default=0x400,
        help="Raw memory search window used when --discover-rich-layout is enabled.",
    )
    parser.add_argument(
        "--layout-discovery-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for rich layout discovery when enabled.",
    )
    parser.add_argument("--max-dx", type=int, default=200, help="Maximum mouse dx per env frame.")
    parser.add_argument("--max-dy", type=int, default=200, help="Maximum mouse dy per env frame.")
    parser.add_argument(
        "--csv",
        type=str,
        default="runs/random_rollout.csv",
        help="CSV output path.",
    )
    add_capture_region_args(parser)
    args = parse_args_allowing_launch_flags(parser)

    if args.reset_backend == RESET_RELAUNCH:
        if args.launch_command is None:
            raise SystemExit("Relaunch rollout requires --launch-command")
        if args.clean_save_path is None or args.active_save_path is None:
            raise SystemExit("Relaunch rollout requires save paths")

    env = GettingOverItEnv(
        pid=args.pid,
        dt=1.0 / args.hz,
        action_repeat=args.action_repeat,
        image_hz=args.image_hz,
        strict_image=args.strict_image,
        capture_region=capture_region_from_args(args),
        enable_uinput=args.send_actions,
        reset_backend=args.reset_backend,
        launch_command=args.launch_command,
        clean_save_path=args.clean_save_path,
        active_save_path=args.active_save_path,
        game_ready_timeout=args.game_ready_timeout,
        startup_mode="auto" if args.startup_mode == "auto" else "legacy",
        startup_title_click=tuple(args.title_click),
        startup_confirm_click=tuple(args.confirm_click),
        window_left=args.window_left,
        window_top=args.window_top,
        window_width=args.window_width,
        window_height=args.window_height,
        discover_rich_layout=args.discover_rich_layout,
        window=args.memory_window,
        layout_discovery_timeout=args.layout_discovery_timeout,
        max_dx=args.max_dx,
        max_dy=args.max_dy,
    )
    rows: list[dict[str, object]] = []
    try:
        obs, info = env.reset()
        started = time.perf_counter()
        reward_total = 0.0
        steps = 0
        terminated = False
        truncated = False
        while time.perf_counter() - started < args.seconds:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            state = obs[STATE_OBS_KEY]
            image = obs[IMAGE_OBS_KEY]
            reward_debug = info.get("reward_debug", {})
            reward_total += float(reward)
            steps += 1
            row = {
                "t": time.perf_counter() - started,
                "step": steps,
                "reward": float(reward),
                "reward_total": reward_total,
                "cursor_x": float(state[0]),
                "cursor_y": float(state[1]),
                "body_y": float(state[5]),
                "progress_y": float(state[12]),
                "best_y": float(state[13]),
                "terminated": terminated,
                "truncated": truncated,
                "rich_state_age": info["rich_state_age"],
                "image_mean": float(image.mean()),
                "image_std": float(image.std()),
                "image_min": int(image.min()),
                "image_max": int(image.max()),
                "image_age": info["image_age"],
                "image_updates": info["image_updates"],
                "active_step_ms": info["step_timing"]["active_step_ms"],
                "wall_step_ms": info["step_timing"]["wall_step_ms"],
                "process_lost": info["process_lost"],
            }
            row.update(
                {
                    "progress_valid": reward_debug.get("progress_valid", False),
                    "progress_source": reward_debug.get("progress_source", "missing"),
                    "progress_y_reward": reward_debug.get("progress_y", 0.0),
                    "reward_delta_y": reward_debug.get("delta_y", 0.0),
                    "reward_delta_best": reward_debug.get("delta_best", 0.0),
                    "reward_best_y": reward_debug.get("best_y", 0.0),
                    "reward_fall": reward_debug.get("fall", False),
                    "reward_reason": reward_debug.get("reward_reason", "missing"),
                    "invalid_steps": reward_debug.get("invalid_steps", 0),
                    "steps_since_progress": reward_debug.get("steps_since_progress", 0),
                }
            )
            rows.append(row)
            if terminated or truncated:
                break
        elapsed = max(1e-9, time.perf_counter() - started)
        _write_csv(args.csv, rows)
        print(f"steps: {steps}")
        print(f"fps: {steps / elapsed:.3f}")
        print(f"reward_total: {reward_total:.6f}")
        reward_std = _reward_std(rows)
        valid_ratio = _progress_valid_ratio(rows)
        source_counts = Counter(str(row.get("progress_source", "missing")) for row in rows)
        reason_counts = Counter(str(row.get("reward_reason", "missing")) for row in rows)
        print(f"reward_std: {reward_std:.9f}")
        print(f"progress_valid_ratio: {valid_ratio:.3f}")
        print(f"progress_sources: {dict(source_counts)}")
        print(f"reward_reasons: {dict(reason_counts)}")
        print(f"reset_mode: {info.get('reset_mode')}")
        print(f"process_lost: {info.get('process_lost')}")
        print(f"body_y: {float(obs[STATE_OBS_KEY][5]):.6f}")
        print(f"best_y: {float(obs[STATE_OBS_KEY][13]):.6f}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        print(f"csv: {args.csv}")
    finally:
        env.close()


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["t", "step", "reward"]
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _reward_std(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float32)
    return float(rewards.std())


def _progress_valid_ratio(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    return sum(bool(row.get("progress_valid", False)) for row in rows) / len(rows)


if __name__ == "__main__":
    main()
