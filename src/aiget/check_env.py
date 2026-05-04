#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np

from .cli_utils import add_capture_region_args, capture_region_from_args
from .env import IMAGE_OBS_KEY, STATE_OBS_KEY, GettingOverItEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Getting Over It Gymnasium env.")
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Target PID. Defaults to the running game.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Random smoke-test steps after reset.",
    )
    parser.add_argument("--hz", type=float, default=30.0, help="Base env frame rate.")
    parser.add_argument("--image-hz", type=float, default=30.0, help="Image capture rate.")
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=2,
        help="Repeated frames per policy action.",
    )
    parser.add_argument(
        "--allow-attach-reset",
        action="store_true",
        help="Allow smoke validation without a real game reset.",
    )
    parser.add_argument(
        "--send-actions",
        action="store_true",
        help="Send random actions through uinput.",
    )
    parser.add_argument(
        "--allow-blank-image",
        action="store_true",
        help="Allow smoke validation to continue with blank/failed image capture.",
    )
    add_capture_region_args(parser)
    args = parser.parse_args()

    if not args.allow_attach_reset:
        raise SystemExit(
            "Refusing check_env: real game reset is not implemented. "
            "Use --allow-attach-reset only for smoke validation."
        )

    try:
        from stable_baselines3.common.env_checker import check_env
    except ModuleNotFoundError as exc:
        raise SystemExit("Install RL extras first: uv sync --extra rl") from exc

    env = GettingOverItEnv(
        pid=args.pid,
        dt=1.0 / args.hz,
        action_repeat=args.action_repeat,
        image_hz=args.image_hz,
        strict_image=not args.allow_blank_image,
        capture_region=capture_region_from_args(args),
        enable_uinput=args.send_actions,
    )
    try:
        check_env(env, warn=True)
        obs, info = env.reset()
        rewards: list[float] = []
        started = time.perf_counter()
        terminated = False
        truncated = False
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            rewards.append(float(reward))
            if terminated or truncated:
                break
        elapsed = max(1e-9, time.perf_counter() - started)
        state = obs[STATE_OBS_KEY]
        image = obs[IMAGE_OBS_KEY]
        print(f"state_shape: {state.shape} dtype={state.dtype}")
        print(f"image_shape: {image.shape} dtype={image.dtype} min={image.min()} max={image.max()}")
        print(f"steps: {len(rewards)}")
        print(f"fps: {len(rewards) / elapsed:.3f}")
        print(f"reward_min: {min(rewards) if rewards else 0.0:.6f}")
        print(f"reward_max: {max(rewards) if rewards else 0.0:.6f}")
        print(f"reward_mean: {float(np.mean(rewards)) if rewards else 0.0:.6f}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        print(f"info: {info}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
