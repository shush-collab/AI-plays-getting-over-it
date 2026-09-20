#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

FAST_VALUE_FIELDS = (
    "cursor_x",
    "cursor_y",
    "cursor_vx",
    "cursor_vy",
)

RICH_VALUE_FIELDS = (
    "body_x",
    "body_y",
    "body_vx",
    "body_vy",
    "hammer_tip_x",
    "hammer_tip_y",
    "hammer_dir_sin",
    "hammer_dir_cos",
    "progress_y",
    "best_progress_y",
    "time_since_progress",
)

ACTION_VALUE_FIELDS = (
    "previous_action_x",
    "previous_action_y",
)

VALUE_FIELDS = FAST_VALUE_FIELDS + RICH_VALUE_FIELDS + ACTION_VALUE_FIELDS
MASK_FIELDS = tuple(f"{name}_valid" for name in FAST_VALUE_FIELDS + RICH_VALUE_FIELDS)
OBSERVATION_FIELDS = VALUE_FIELDS + MASK_FIELDS

FAST_DIM = len(FAST_VALUE_FIELDS)
RICH_DIM = len(RICH_VALUE_FIELDS)
ACTION_DIM = len(ACTION_VALUE_FIELDS)
VALUE_DIM = len(VALUE_FIELDS)
MASK_DIM = len(MASK_FIELDS)
OBS_DIM = len(OBSERVATION_FIELDS)

IDX_CURSOR = slice(0, FAST_DIM)
IDX_RICH = slice(FAST_DIM, FAST_DIM + RICH_DIM)
IDX_PREVIOUS_ACTION = slice(FAST_DIM + RICH_DIM, VALUE_DIM)
IDX_MASK = slice(VALUE_DIM, OBS_DIM)
IDX_FAST_MASK = slice(VALUE_DIM, VALUE_DIM + FAST_DIM)
IDX_RICH_MASK = slice(VALUE_DIM + FAST_DIM, OBS_DIM)


def build_observation_vector(
    cursor_xy: Sequence[float],
    cursor_vxy: Sequence[float],
    latest_rich: object,
    previous_action: Sequence[float],
    out: np.ndarray,
) -> np.ndarray:
    if out.shape != (OBS_DIM,):
        raise ValueError(f"out must have shape ({OBS_DIM},), got {out.shape}")

    out.fill(0.0)
    out[0] = float(cursor_xy[0])
    out[1] = float(cursor_xy[1])
    out[2] = float(cursor_vxy[0])
    out[3] = float(cursor_vxy[1])
    out[IDX_PREVIOUS_ACTION] = np.asarray(previous_action, dtype=np.float32)[:ACTION_DIM]

    rich_values, rich_mask = _coerce_rich_arrays(latest_rich)
    if rich_values is not None:
        if rich_mask is None:
            out[IDX_RICH] = rich_values[:RICH_DIM]
            out[IDX_RICH_MASK] = 1.0
        else:
            mask = rich_mask[:RICH_DIM]
            out[IDX_RICH] = rich_values[:RICH_DIM] * mask
            out[IDX_RICH_MASK] = mask

    out[IDX_FAST_MASK] = 1.0
    return out


def empty_rich_arrays() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(RICH_DIM, dtype=np.float32), np.zeros(RICH_DIM, dtype=np.float32)


def _coerce_rich_arrays(latest_rich: object) -> tuple[np.ndarray | None, np.ndarray | None]:
    if latest_rich is None:
        return None, None
    if isinstance(latest_rich, tuple) and len(latest_rich) == 2:
        return (
            np.asarray(latest_rich[0], dtype=np.float32),
            np.asarray(latest_rich[1], dtype=np.float32),
        )
    if isinstance(latest_rich, Mapping):
        values_obj = latest_rich.get("values")
        mask_obj = latest_rich.get("mask")
        if values_obj is not None:
            return (
                np.asarray(values_obj, dtype=np.float32),
                None if mask_obj is None else np.asarray(mask_obj, dtype=np.float32),
            )
        values = np.zeros(RICH_DIM, dtype=np.float32)
        mask = np.zeros(RICH_DIM, dtype=np.float32)
        for index, field_name in enumerate(RICH_VALUE_FIELDS):
            if field_name in latest_rich:
                values[index] = float(latest_rich[field_name])
                mask[index] = 1.0
        valid_mask = latest_rich.get("valid_mask")
        if isinstance(valid_mask, Mapping):
            for index, field_name in enumerate(RICH_VALUE_FIELDS):
                mask[index] = 1.0 if valid_mask.get(field_name, bool(mask[index])) else 0.0
        return values, mask
    values = getattr(latest_rich, "values", None)
    mask = getattr(latest_rich, "mask", None)
    if values is not None:
        return (
            np.asarray(values, dtype=np.float32),
            None if mask is None else np.asarray(mask, dtype=np.float32),
        )
    return None, None
