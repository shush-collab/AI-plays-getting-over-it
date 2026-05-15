#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cli_utils import (
    add_capture_region_args,
    capture_region_from_args,
    parse_args_allowing_launch_flags,
)
from .env import RESET_ATTACH, RESET_RELAUNCH, GettingOverItEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small SB3 MultiInputPolicy agent for Getting Over It."
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Target PID. Defaults to the running game.",
    )
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac", help="SB3 algorithm.")
    parser.add_argument("--steps", type=int, default=10_000, help="Training timesteps.")
    parser.add_argument("--hz", type=float, default=30.0, help="Base env frame rate.")
    parser.add_argument("--image-hz", type=float, default=30.0, help="Image capture rate.")
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=2,
        help="Repeated frames per policy action.",
    )
    parser.add_argument("--tensorboard-log", type=str, default="runs/", help="TensorBoard log dir.")
    parser.add_argument(
        "--model-out",
        type=str,
        default="runs/aiget_sac_v1",
        help="Saved model path.",
    )
    parser.add_argument(
        "--allow-attach-reset",
        action="store_true",
        help="Allow training with attach-only reset. Not recommended for real runs.",
    )
    parser.add_argument(
        "--reset-backend",
        choices=(RESET_ATTACH, RESET_RELAUNCH),
        default=RESET_ATTACH,
        help="Game reset backend.",
    )
    parser.add_argument(
        "--launch-command",
        nargs="+",
        default=None,
        help="Command used to launch the game for relaunch reset.",
    )
    parser.add_argument("--clean-save-path", type=str, default=None, help="Known clean save path.")
    parser.add_argument("--active-save-path", type=str, default=None, help="Runtime save path.")
    parser.add_argument("--game-ready-timeout", type=float, default=45.0)
    parser.add_argument("--startup-mode", choices=("legacy", "auto"), default="auto")
    parser.add_argument("--title-click", nargs=2, type=int, default=(1275, 305))
    parser.add_argument("--confirm-click", nargs=2, type=int, default=(1275, 305))
    parser.add_argument("--window-left", type=int, default=320)
    parser.add_argument("--window-top", type=int, default=178)
    parser.add_argument("--window-width", type=int, default=1920)
    parser.add_argument("--window-height", type=int, default=1080)
    parser.add_argument(
        "--preflight-steps",
        type=int,
        default=100,
        help="Random steps used to check reward variance before learning.",
    )
    parser.add_argument(
        "--min-reward-std",
        type=float,
        default=1e-4,
        help="Minimum reward standard deviation required before training.",
    )
    parser.add_argument(
        "--min-progress-valid-ratio",
        type=float,
        default=0.8,
        help="Minimum fraction of preflight steps with valid progress.",
    )
    parser.add_argument(
        "--send-actions",
        action="store_true",
        help="Send policy actions through uinput.",
    )
    parser.add_argument(
        "--memory-window",
        type=lambda value: int(value, 0),
        default=0x400,
        help="Raw memory search window for resolving reward progress fields.",
    )
    parser.add_argument(
        "--layout-discovery-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for resolving reward progress memory fields.",
    )
    parser.add_argument("--max-dx", type=int, default=200, help="Maximum mouse dx per env frame.")
    parser.add_argument("--max-dy", type=int, default=200, help="Maximum mouse dy per env frame.")
    add_capture_region_args(parser)
    args = parse_args_allowing_launch_flags(parser)

    if args.reset_backend == RESET_ATTACH and not args.allow_attach_reset:
        raise SystemExit(
            "Refusing to train: real game reset is not implemented. "
            "Use --reset-backend relaunch with save restore for real training, "
            "or --allow-attach-reset only for smoke tests."
        )
    capture_region = capture_region_from_args(args)
    if capture_region is None:
        raise SystemExit("Training requires explicit game capture region.")
    if args.reset_backend == RESET_RELAUNCH:
        if args.launch_command is None:
            raise SystemExit("Relaunch reset requires --launch-command.")
        if args.clean_save_path is None or args.active_save_path is None:
            raise SystemExit("Relaunch reset requires --clean-save-path and --active-save-path.")

    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        raise SystemExit("Install RL extras first: uv sync --extra rl") from exc

    env = GettingOverItEnv(
        pid=args.pid,
        dt=1.0 / args.hz,
        action_repeat=args.action_repeat,
        image_hz=args.image_hz,
        strict_image=True,
        capture_region=capture_region,
        reset_backend=args.reset_backend,
        launch_command=args.launch_command,
        clean_save_path=args.clean_save_path,
        active_save_path=args.active_save_path,
        game_ready_timeout=args.game_ready_timeout,
        startup_mode=args.startup_mode,
        startup_title_click=tuple(args.title_click),
        startup_confirm_click=tuple(args.confirm_click),
        window_left=args.window_left,
        window_top=args.window_top,
        window_width=args.window_width,
        window_height=args.window_height,
        enable_uinput=args.send_actions,
        discover_rich_layout=True,
        window=args.memory_window,
        layout_discovery_timeout=args.layout_discovery_timeout,
        max_dx=args.max_dx,
        max_dy=args.max_dy,
    )
    monitor_path = Path(args.tensorboard_log) / "monitor.csv"
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    env = Monitor(
        env,
        filename=str(monitor_path),
        info_keywords=(
            "progress_y",
            "progress_valid",
            "progress_source",
            "reward_reason",
            "process_lost",
            "reset_mode",
        ),
    )
    if args.preflight_steps > 0:
        _run_reward_preflight(
            env,
            args.preflight_steps,
            args.min_reward_std,
            args.min_progress_valid_ratio,
        )
    algo_cls = SAC if args.algo == "sac" else PPO
    model = algo_cls(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
    )
    try:
        model.learn(total_timesteps=args.steps)
        model_path = Path(args.model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print(f"saved_model: {model_path}")
    finally:
        env.close()


def _run_reward_preflight(
    env: GettingOverItEnv,
    steps: int,
    min_reward_std: float,
    min_progress_valid_ratio: float,
) -> None:
    rewards: list[float] = []
    valid_count = 0
    source_counts: dict[str, int] = {}

    env.reset()

    for _ in range(steps):
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        rewards.append(float(reward))

        rd = info.get("reward_debug", {})
        if rd.get("progress_valid", False):
            valid_count += 1

        source = str(rd.get("progress_source", "missing"))
        source_counts[source] = source_counts.get(source, 0) + 1

        if terminated or truncated:
            env.reset()

    reward_std = float(np.asarray(rewards, dtype=np.float32).std()) if rewards else 0.0
    valid_ratio = valid_count / max(1, len(rewards))

    print(f"preflight_reward_std: {reward_std:.9f}")
    print(f"preflight_progress_valid_ratio: {valid_ratio:.3f}")
    print(f"preflight_progress_sources: {source_counts}")

    if reward_std <= min_reward_std:
        raise SystemExit("Refusing to train: reward_std is too low.")

    if valid_ratio < min_progress_valid_ratio:
        raise SystemExit("Refusing to train: progress signal is invalid too often.")

    if source_counts.get("invalid", 0) > len(rewards) * 0.2:
        raise SystemExit("Refusing to train: too many invalid progress samples.")

    env.reset()


if __name__ == "__main__":
    main()
