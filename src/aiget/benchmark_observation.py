#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np

from .cli_utils import add_capture_region_args, capture_region_from_args
from .env import GettingOverItEnv


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    env = GettingOverItEnv(
        pid=args.pid,
        dt=1.0 / args.hz,
        action_repeat=args.action_repeat,
        image_hz=args.image_hz,
        strict_image=args.strict_image,
        capture_region=capture_region_from_args(args),
        rich_snapshot_interval=args.rich_snapshot_interval,
        enable_uinput=args.send_actions,
        live_layout_cache=args.live_layout_cache,
        use_layout_cache=not args.no_live_layout_cache,
        discover_rich_layout=args.discover_rich_layout,
    )
    active_times_ms: list[float] = []
    wall_times_ms: list[float] = []
    sleep_times_ms: list[float] = []
    missed_deadlines = 0
    action = np.zeros(2, dtype=np.float32)
    startup_started = time.monotonic()
    try:
        _, info = env.reset()
        startup_ms = (time.monotonic() - startup_started) * 1000.0
        started = time.monotonic()
        start_rich_updates = int(info["rich_updates"])
        while time.monotonic() - started < args.seconds:
            _, _, _, _, info = env.step(action)
            timing = info["step_timing"]
            active_times_ms.append(float(timing["active_step_ms"]))
            wall_times_ms.append(float(timing["wall_step_ms"]))
            sleep_times_ms.append(float(timing["sleep_ms"]))
            missed_deadlines += int(timing["missed_deadlines"])
        elapsed = max(1e-9, time.monotonic() - started)
        rich_updates = int(info["rich_updates"]) - start_rich_updates
        active = np.asarray(active_times_ms, dtype=np.float32)
        wall = np.asarray(wall_times_ms, dtype=np.float32)
        sleep = np.asarray(sleep_times_ms, dtype=np.float32)
        return {
            "policy_hz": len(wall_times_ms) / elapsed,
            "fast_hz": len(wall_times_ms) * args.action_repeat / elapsed,
            "rich_hz": rich_updates / elapsed,
            "avg_active_step_ms": float(active.mean()) if active.size else 0.0,
            "p95_active_step_ms": float(np.percentile(active, 95)) if active.size else 0.0,
            "avg_wall_step_ms": float(wall.mean()) if wall.size else 0.0,
            "p95_wall_step_ms": float(np.percentile(wall, 95)) if wall.size else 0.0,
            "avg_sleep_ms": float(sleep.mean()) if sleep.size else 0.0,
            "missed_deadlines": missed_deadlines,
            "startup_ms": startup_ms,
            "dropped_rich_updates": info["dropped_rich_updates"],
            "image_hz": info["image_updates"] / elapsed,
            "image_age": info["image_age"],
            "image_std": info["image_std"],
            "game_freeze_detected": "yes" if info["game_freeze_detected"] else "no",
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Getting Over It observation env latency."
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Target PID. Defaults to the running game.",
    )
    parser.add_argument("--seconds", type=float, default=10.0, help="Benchmark duration.")
    parser.add_argument("--hz", type=float, default=30.0, help="Env step rate target.")
    parser.add_argument("--image-hz", type=float, default=30.0, help="Image capture rate.")
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=2,
        help="Repeated frames per policy action.",
    )
    parser.add_argument(
        "--rich-snapshot-interval",
        type=float,
        default=0.2,
        help="Background rich refresh interval.",
    )
    parser.add_argument(
        "--live-layout-cache",
        type=str,
        default=None,
        help="Optional layout cache path.",
    )
    parser.add_argument(
        "--no-live-layout-cache",
        action="store_true",
        help="Disable layout cache.",
    )
    parser.add_argument(
        "--send-actions",
        action="store_true",
        help="Send zero mouse actions through uinput.",
    )
    parser.add_argument(
        "--discover-rich-layout",
        action="store_true",
        help="Run optional rich layout discovery during reset instead of using cache/fast-only.",
    )
    parser.add_argument(
        "--strict-image",
        action="store_true",
        help="Fail if image capture is blank, stale, or unavailable.",
    )
    add_capture_region_args(parser)
    args = parser.parse_args()

    result = run_benchmark(args)
    for key in (
        "fast_hz",
        "policy_hz",
        "rich_hz",
        "avg_active_step_ms",
        "p95_active_step_ms",
        "avg_wall_step_ms",
        "p95_wall_step_ms",
        "avg_sleep_ms",
        "missed_deadlines",
        "startup_ms",
        "dropped_rich_updates",
        "image_hz",
        "image_age",
        "image_std",
        "game_freeze_detected",
    ):
        value = result[key]
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
