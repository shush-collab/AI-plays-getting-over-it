#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .env import GettingOverItEnv


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
        "--send-actions",
        action="store_true",
        help="Send policy actions through uinput.",
    )
    args = parser.parse_args()

    if not args.allow_attach_reset:
        raise SystemExit(
            "Refusing to train: real game reset is not implemented. "
            "Run check_env/random_rollout first, then use --allow-attach-reset "
            "only for smoke tests."
        )

    try:
        from stable_baselines3 import PPO, SAC
    except ModuleNotFoundError as exc:
        raise SystemExit("Install RL extras first: uv sync --extra rl") from exc

    env = GettingOverItEnv(
        pid=args.pid,
        dt=1.0 / args.hz,
        action_repeat=args.action_repeat,
        enable_uinput=args.send_actions,
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


if __name__ == "__main__":
    main()
