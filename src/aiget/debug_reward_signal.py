#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from .cli_utils import add_capture_region_args, capture_region_from_args
from .env import IMAGE_OBS_KEY, STATE_OBS_KEY, GettingOverItEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--send-actions", action="store_true")
    parser.add_argument("--csv", default="runs/reward_signal.csv")
    parser.add_argument("--min-valid-ratio", type=float, default=0.8)
    parser.add_argument("--min-reward-std", type=float, default=1e-4)
    parser.add_argument("--memory-window", type=lambda value: int(value, 0), default=0x400)
    parser.add_argument("--layout-discovery-timeout", type=float, default=5.0)
    parser.add_argument("--max-dx", type=int, default=200)
    parser.add_argument("--max-dy", type=int, default=200)
    add_capture_region_args(parser)
    args = parser.parse_args()

    env = GettingOverItEnv(
        capture_region=capture_region_from_args(args),
        strict_image=True,
        enable_uinput=args.send_actions,
        discover_rich_layout=True,
        window=args.memory_window,
        layout_discovery_timeout=args.layout_discovery_timeout,
        max_dx=args.max_dx,
        max_dy=args.max_dy,
    )

    rows: list[dict[str, object]] = []
    try:
        obs, info = env.reset()
        start = time.perf_counter()

        while time.perf_counter() - start < args.seconds:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            state = obs[STATE_OBS_KEY]
            image = obs[IMAGE_OBS_KEY]
            rd = info.get("reward_debug", {})

            rows.append(
                {
                    "t": time.perf_counter() - start,
                    "reward": float(reward),
                    "cursor_x": float(state[0]),
                    "cursor_y": float(state[1]),
                    "body_y": float(state[5]),
                    "body_valid": bool(state[22]),
                    "progress_y": float(state[12]),
                    "progress_valid_mask": bool(state[29]),
                    "progress_y_reward": rd.get("progress_y", 0.0),
                    "progress_valid": rd.get("progress_valid", False),
                    "progress_source": rd.get("progress_source", "missing"),
                    "delta_y": rd.get("delta_y", 0.0),
                    "delta_best": rd.get("delta_best", 0.0),
                    "best_y": rd.get("best_y", 0.0),
                    "reward_reason": rd.get("reward_reason", "missing"),
                    "image_mean": float(image.mean()),
                    "image_std": float(image.std()),
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )

            if terminated or truncated:
                break

        write_csv(args.csv, rows)

        rewards = np.asarray([float(r["reward"]) for r in rows], dtype=np.float32)
        valid_ratio = sum(bool(r["progress_valid"]) for r in rows) / max(1, len(rows))
        reward_std = float(rewards.std()) if len(rewards) else 0.0

        print(f"rows: {len(rows)}")
        print(f"reward_std: {reward_std:.9f}")
        print(f"progress_valid_ratio: {valid_ratio:.3f}")
        print(f"csv: {args.csv}")

        if valid_ratio < args.min_valid_ratio:
            raise SystemExit("FAIL: progress signal valid ratio too low")

        if reward_std <= args.min_reward_std:
            raise SystemExit("FAIL: reward_std too low")

    finally:
        env.close()


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("")
        return
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
