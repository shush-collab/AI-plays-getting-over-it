from __future__ import annotations

from dataclasses import dataclass

from .progress_signal import ProgressSample


@dataclass
class RewardState:
    last_y: float = 0.0
    best_y: float = 0.0
    initialized: bool = False
    steps_since_progress: int = 0
    invalid_steps: int = 0


@dataclass(frozen=True)
class RewardOutput:
    reward: float
    terminated: bool
    truncated: bool
    debug: dict[str, float | int | str | bool]


class HeightReward:
    def __init__(
        self,
        step_penalty: float = 0.001,
        delta_scale: float = 0.1,
        best_scale: float = 2.0,
        fall_threshold: float = 5.0,
        fall_penalty: float = 1.0,
        invalid_penalty: float = 0.002,
        no_progress_limit_steps: int = 900,
        max_invalid_steps: int = 90,
        reward_clip: float = 5.0,
    ):
        self.step_penalty = step_penalty
        self.delta_scale = delta_scale
        self.best_scale = best_scale
        self.fall_threshold = fall_threshold
        self.fall_penalty = fall_penalty
        self.invalid_penalty = invalid_penalty
        self.no_progress_limit_steps = no_progress_limit_steps
        self.max_invalid_steps = max_invalid_steps
        self.reward_clip = reward_clip
        self.state = RewardState()

    def reset(self, initial_progress: ProgressSample | None = None) -> None:
        self.state = RewardState()
        if initial_progress is not None and initial_progress.valid:
            self.state.last_y = initial_progress.y
            self.state.best_y = initial_progress.y
            self.state.initialized = True

    def compute(self, progress: ProgressSample, time_limit_reached: bool = False) -> RewardOutput:
        s = self.state

        if not progress.valid:
            s.invalid_steps += 1
            reward = -self.invalid_penalty
            truncated = time_limit_reached or s.invalid_steps >= self.max_invalid_steps
            return RewardOutput(
                reward=reward,
                terminated=False,
                truncated=truncated,
                debug={
                    "progress_valid": False,
                    "progress_source": progress.source.value,
                    "progress_y": progress.y,
                    "last_y": s.last_y,
                    "best_y": s.best_y,
                    "invalid_steps": s.invalid_steps,
                    "steps_since_progress": s.steps_since_progress,
                    "reward_reason": "invalid_progress",
                },
            )

        s.invalid_steps = 0

        if not s.initialized:
            s.last_y = progress.y
            s.best_y = progress.y
            s.initialized = True
            return RewardOutput(
                reward=-self.step_penalty,
                terminated=False,
                truncated=time_limit_reached,
                debug={
                    "progress_valid": True,
                    "progress_source": progress.source.value,
                    "progress_y": progress.y,
                    "last_y": s.last_y,
                    "best_y": s.best_y,
                    "reward_reason": "initialization",
                },
            )

        delta_y = progress.y - s.last_y
        previous_best = s.best_y
        delta_best = max(0.0, progress.y - previous_best)

        if delta_best > 0:
            s.best_y = progress.y
            s.steps_since_progress = 0
        else:
            s.steps_since_progress += 1

        fall = progress.y < s.best_y - self.fall_threshold

        reward = (
            self.best_scale * delta_best
            + self.delta_scale * delta_y
            - self.step_penalty
            - (self.fall_penalty if fall else 0.0)
        )

        reward = max(-self.reward_clip, min(self.reward_clip, reward))

        s.last_y = progress.y

        truncated = time_limit_reached or s.steps_since_progress >= self.no_progress_limit_steps

        return RewardOutput(
            reward=reward,
            terminated=False,
            truncated=truncated,
            debug={
                "progress_valid": True,
                "progress_source": progress.source.value,
                "progress_y": progress.y,
                "last_y": s.last_y,
                "best_y": s.best_y,
                "delta_y": delta_y,
                "delta_best": delta_best,
                "fall": fall,
                "invalid_steps": s.invalid_steps,
                "steps_since_progress": s.steps_since_progress,
                "reward_reason": "height_progress",
            },
        )
