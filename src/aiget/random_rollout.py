#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from .env import STATE_OBS_KEY, GettingOverItEnv


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
        "--csv",
        type=str,
        default="runs/random_rollout.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    env = GettingOverItEnv(
        pid=args.pid,
        dt=1.0 / args.hz,
        action_repeat=args.action_repeat,
        enable_uinput=args.send_actions,
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
            reward_total += float(reward)
            steps += 1
            rows.append(
                {
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
                    "active_step_ms": info["step_timing"]["active_step_ms"],
                    "wall_step_ms": info["step_timing"]["wall_step_ms"],
                }
            )
            if terminated or truncated:
                break
        elapsed = max(1e-9, time.perf_counter() - started)
        _write_csv(args.csv, rows)
        print(f"steps: {steps}")
        print(f"fps: {steps / elapsed:.3f}")
        print(f"reward_total: {reward_total:.6f}")
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


if __name__ == "__main__":
    main()
