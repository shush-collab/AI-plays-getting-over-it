#!/usr/bin/env python3
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .png_utils import read_png, to_gray
from .window_control import WindowControl


class StartupState(Enum):
    UNKNOWN = "UNKNOWN"
    DARK_LOADING = "DARK_LOADING"
    TITLE_MENU = "TITLE_MENU"
    CONFIRM_DIALOG = "CONFIRM_DIALOG"
    SETTINGS_MENU = "SETTINGS_MENU"
    PLAYABLE = "PLAYABLE"
    FAILED = "FAILED"


@dataclass
class StartupEvent:
    index: int
    elapsed_ms: float
    state: StartupState
    action: str
    frame: np.ndarray
    after_frame: np.ndarray | None = None

    def as_trace(self) -> dict[str, object]:
        return {
            "index": self.index,
            "elapsed_ms": self.elapsed_ms,
            "state": self.state.value,
            "action": self.action,
        }


@dataclass
class StartupResult:
    success: bool
    state: StartupState
    elapsed_ms: float
    attempts: int
    layout: Any | None = None
    reason: str = ""
    events: list[StartupEvent] = field(default_factory=list)

    def as_trace(self) -> dict[str, object]:
        return {
            "success": self.success,
            "state": self.state.value,
            "elapsed_ms": self.elapsed_ms,
            "attempts": self.attempts,
            "reason": self.reason,
            "events": [event.as_trace() for event in self.events],
        }


FrameGetter = Callable[[], np.ndarray]
LayoutResolver = Callable[[], Any]


class StartupAutomation:
    def __init__(
        self,
        *,
        frame_getter: FrameGetter,
        window_control: WindowControl,
        resolve_layout: LayoutResolver,
        timeout: float,
        reference_dir: str | Path | None = None,
        probe_interval: float = 0.5,
        action_interval: float = 1.0,
        max_actions: int = 2,
        title_click: tuple[int, int] = (850, 210),
        confirm_click: tuple[int, int] = (640, 430),
        full_frame_getter: FrameGetter | None = None,
    ):
        self.frame_getter = frame_getter
        self.full_frame_getter = full_frame_getter or frame_getter
        self.window_control = window_control
        self.resolve_layout = resolve_layout
        self.timeout = timeout
        self.probe_interval = probe_interval
        self.action_interval = action_interval
        self.max_actions = max_actions
        self.title_click = title_click
        self.confirm_click = confirm_click
        self.references = load_reference_frames(reference_dir)

    def drive_until_playable(self) -> StartupResult:
        started = time.perf_counter()
        deadline = time.monotonic() + self.timeout
        events: list[StartupEvent] = []
        last_state = StartupState.UNKNOWN
        last_reason = "PLAYERCONTROL_TIMEOUT"
        actions = 0

        initial = self._wait_for_nonblank_frame(deadline)
        if initial is None:
            return StartupResult(
                success=False,
                state=StartupState.DARK_LOADING,
                elapsed_ms=_elapsed_ms(started),
                attempts=0,
                reason="IMAGE_DARK_TIMEOUT",
                events=events,
            )

        layout = self._probe_until(time.monotonic() + self.probe_interval)
        if layout is not None:
            elapsed = _elapsed_ms(started)
            events.append(_event(len(events), elapsed, StartupState.PLAYABLE, "resolved", initial))
            return StartupResult(
                success=True,
                state=StartupState.PLAYABLE,
                elapsed_ms=elapsed,
                attempts=0,
                layout=layout,
                events=events,
            )

        click_sequence = [
            ("click_primary", self.title_click),
            ("click_fallback", self.confirm_click),
        ][: self.max_actions]
        for action_name, click in click_sequence:
            if time.monotonic() >= deadline:
                break
            before = self.full_frame_getter()
            state = classify_startup_frame(before, self.references)
            last_state = state
            self.window_control.click_relative(*click)
            actions += 1
            time.sleep(min(0.25, self.action_interval))
            after = self.full_frame_getter()
            event = _event(
                len(events),
                _elapsed_ms(started),
                state,
                f"{action_name}:{click[0]},{click[1]}",
                before,
            )
            event.after_frame = _gray(after).copy()
            events.append(event)

            layout = self._probe_until(min(deadline, time.monotonic() + self.action_interval))
            if layout is not None:
                elapsed = _elapsed_ms(started)
                events.append(
                    _event(len(events), elapsed, StartupState.PLAYABLE, "resolved", after)
                )
                return StartupResult(
                    success=True,
                    state=StartupState.PLAYABLE,
                    elapsed_ms=elapsed,
                    attempts=actions,
                    layout=layout,
                    events=events,
                )

        return StartupResult(
            success=False,
            state=last_state,
            elapsed_ms=_elapsed_ms(started),
            attempts=actions,
            reason=last_reason,
            events=events,
        )

    def _wait_for_nonblank_frame(self, deadline: float) -> np.ndarray | None:
        while time.monotonic() < deadline:
            frame = self.full_frame_getter()
            state = classify_startup_frame(frame, self.references)
            if state != StartupState.DARK_LOADING:
                return frame
            time.sleep(0.1)
        return None

    def _probe_until(self, deadline: float) -> Any | None:
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self.resolve_layout()
            except Exception as exc:
                last_exc = exc
                time.sleep(self.probe_interval)
        if last_exc is not None:
            raise_last_reason = _stage_from_exception(last_exc)
            if raise_last_reason == "FAST_CURSOR_TIMEOUT":
                raise RuntimeError(raise_last_reason) from last_exc
        return None


def classify_startup_frame(
    frame: np.ndarray,
    references: dict[StartupState, np.ndarray] | None = None,
) -> StartupState:
    gray = _gray(frame)
    mean = float(gray.mean())
    std = float(gray.std())
    if mean < 3.0 or std < 2.0:
        return StartupState.DARK_LOADING

    if references:
        state, score = _nearest_reference(gray, references)
        if score < 25.0:
            return state

    bright_ratio = float((gray > 180).mean())
    if bright_ratio < 0.001:
        return StartupState.DARK_LOADING
    if std > 18.0:
        return StartupState.TITLE_MENU
    return StartupState.UNKNOWN


def load_reference_frames(reference_dir: str | Path | None) -> dict[StartupState, np.ndarray]:
    if reference_dir is None:
        return {}
    base = Path(reference_dir).expanduser()
    if not base.exists():
        return {}
    mapping = {
        StartupState.DARK_LOADING: "dark_loading.png",
        StartupState.TITLE_MENU: "title_continue.png",
        StartupState.CONFIRM_DIALOG: "confirm_dialog.png",
        StartupState.SETTINGS_MENU: "settings.png",
        StartupState.PLAYABLE: "playable.png",
    }
    refs: dict[StartupState, np.ndarray] = {}
    for state, filename in mapping.items():
        path = base / filename
        if path.exists():
            try:
                refs[state] = to_gray(read_png(path))
            except Exception:
                continue
    title_new = base / "title_new_game.png"
    if title_new.exists() and StartupState.TITLE_MENU not in refs:
        try:
            refs[StartupState.TITLE_MENU] = to_gray(read_png(title_new))
        except Exception:
            pass
    return refs


def _nearest_reference(
    gray: np.ndarray,
    references: dict[StartupState, np.ndarray],
) -> tuple[StartupState, float]:
    best_state = StartupState.UNKNOWN
    best_score = float("inf")
    for state, ref in references.items():
        if ref.shape != gray.shape:
            ref = _resize_nearest(ref, gray.shape[0], gray.shape[1])
        score = float(np.mean(np.abs(gray.astype(np.int16) - ref.astype(np.int16))))
        if score < best_score:
            best_state = state
            best_score = score
    return best_state, best_score


def _stage_from_exception(exc: Exception) -> str:
    text = str(exc)
    if "FindObjectOfType(PlayerControl)" in text:
        return "PLAYERCONTROL_TIMEOUT"
    if "fakeCursorRB" in text:
        return "FAST_CURSOR_TIMEOUT"
    return "PLAYERCONTROL_TIMEOUT"


def _event(
    index: int,
    elapsed_ms: float,
    state: StartupState,
    action: str,
    frame: np.ndarray,
) -> StartupEvent:
    return StartupEvent(
        index=index,
        elapsed_ms=elapsed_ms,
        state=state,
        action=action,
        frame=_gray(frame).copy(),
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _gray(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim == 3:
        return array[:, :, -1].astype(np.uint8, copy=False)
    raise ValueError(f"expected 2D or 3D frame, got shape {array.shape}")


def _resize_nearest(gray: np.ndarray, height: int, width: int) -> np.ndarray:
    y_idx = np.linspace(0, gray.shape[0] - 1, height).astype(np.intp)
    x_idx = np.linspace(0, gray.shape[1] - 1, width).astype(np.intp)
    return gray[y_idx][:, x_idx]
