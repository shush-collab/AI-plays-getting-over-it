from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ProgressSource(str, Enum):  # noqa: UP042
    MEMORY_BODY = "memory_body"
    MEMORY_PROGRESS = "memory_progress"
    VISION_TEMPLATE = "vision_template"
    INVALID = "invalid"


@dataclass(frozen=True)
class ProgressSample:
    y: float
    valid: bool
    source: ProgressSource
    confidence: float
    debug: dict[str, float | str | bool]


class ProgressEstimator:
    """
    Converts state/image into one scalar progress_y.

    Priority:
    1. memory body_y
    2. memory progress_y
    3. vision/template progress
    4. invalid
    """

    def __init__(self, vision_estimator: object | None = None):
        self.vision_estimator = vision_estimator

    def estimate(self, state: np.ndarray, image: np.ndarray | None = None) -> ProgressSample:
        body_y = float(state[5])
        progress_y = float(state[12])

        body_valid = bool(state[22])
        progress_valid = bool(state[29])

        if body_valid and np.isfinite(body_y):
            return ProgressSample(
                y=body_y,
                valid=True,
                source=ProgressSource.MEMORY_BODY,
                confidence=1.0,
                debug={
                    "body_y": body_y,
                    "progress_y": progress_y,
                    "body_valid": True,
                    "progress_valid": progress_valid,
                },
            )

        if progress_valid and np.isfinite(progress_y):
            return ProgressSample(
                y=progress_y,
                valid=True,
                source=ProgressSource.MEMORY_PROGRESS,
                confidence=1.0,
                debug={
                    "body_y": body_y,
                    "progress_y": progress_y,
                    "body_valid": False,
                    "progress_valid": True,
                },
            )

        if self.vision_estimator is not None and image is not None:
            sample = self.vision_estimator.estimate(image)
            if sample.valid:
                return sample

        return ProgressSample(
            y=0.0,
            valid=False,
            source=ProgressSource.INVALID,
            confidence=0.0,
            debug={
                "body_y": body_y,
                "progress_y": progress_y,
                "body_valid": body_valid,
                "progress_valid": progress_valid,
            },
        )
