#!/usr/bin/env python3
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ...shared.png_utils import read_png, to_gray
from .window_control import WindowControl


class StartupState(Enum):
    UNKNOWN = "UNKNOWN"
    DARK_LOADING = "DARK_LOADING"
    TITLE_BACKGROUND = "TITLE_BACKGROUND"
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
    frame_change_score: float | None = None
    input_backend: str = ""
    error: str = ""

    def as_trace(self) -> dict[str, object]:
        trace: dict[str, object] = {
            "index": self.index,
            "elapsed_ms": self.elapsed_ms,
            "state": self.state.value,
            "action": self.action,
        }
        if self.frame_change_score is not None:
            trace["frame_change_score"] = self.frame_change_score
        if self.input_backend:
            trace["input_backend"] = self.input_backend
        if self.error:
            trace["error"] = self.error
        return trace


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
        startup_key: str | None = None,
        frame_change_threshold: float = 2.0,
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
        self.startup_key = startup_key
        self.frame_change_threshold = frame_change_threshold
        self.references = load_reference_frames(reference_dir)

    def drive_until_playable(self) -> StartupResult:
        started = time.perf_counter()
        deadline = time.monotonic() + self.timeout
        events: list[StartupEvent] = []
        last_state = StartupState.UNKNOWN
        last_reason = "PLAYERCONTROL_TIMEOUT"
        actions = 0

        initial, wait_state = self._wait_for_actionable_frame(deadline)
        if initial is None:
            return StartupResult(
                success=False,
                state=wait_state,
                elapsed_ms=_elapsed_ms(started),
                attempts=0,
                reason="MENU_VISIBLE_TIMEOUT",
                events=events,
            )

        initial_state = classify_startup_frame(initial, self.references)
        last_state = initial_state
        if initial_state == StartupState.PLAYABLE:
            layout = self._probe_until(time.monotonic() + self.probe_interval)
            if layout is not None:
                elapsed = _elapsed_ms(started)
                events.append(
                    _event(len(events), elapsed, StartupState.PLAYABLE, "resolved", initial)
                )
                return StartupResult(
                    success=True,
                    state=StartupState.PLAYABLE,
                    elapsed_ms=elapsed,
                    attempts=0,
                    layout=layout,
                    events=events,
                )

        actions_to_try = self._startup_actions()
        frame_changed = False
        current_frame = initial
        for action_name, payload in actions_to_try:
            if time.monotonic() >= deadline:
                break
            before = current_frame
            state = classify_startup_frame(before, self.references)
            last_state = state
            if state == StartupState.PLAYABLE:
                layout = self._probe_until(min(deadline, time.monotonic() + self.probe_interval))
                if layout is not None:
                    elapsed = _elapsed_ms(started)
                    events.append(
                        _event(len(events), elapsed, StartupState.PLAYABLE, "resolved", before)
                    )
                    return StartupResult(
                        success=True,
                        state=StartupState.PLAYABLE,
                        elapsed_ms=elapsed,
                        attempts=actions,
                        layout=layout,
                        events=events,
                    )
            if not _state_accepts_startup_input(state):
                current_frame, wait_state = self._wait_for_actionable_frame(deadline)
                if current_frame is None:
                    last_state = wait_state
                    break
                continue
            error = ""
            input_backend = ""
            try:
                if action_name.startswith("click"):
                    assert isinstance(payload, tuple)
                    self.window_control.click_relative(*payload)
                else:
                    assert isinstance(payload, str)
                    self.window_control.key(payload)
                input_backend = self.window_control.active_input_backend
                actions += 1
            except Exception as exc:
                error = str(exc)
            time.sleep(min(0.25, self.action_interval))
            after = self.full_frame_getter()
            change_score = _frame_change_score(before, after)
            frame_changed = frame_changed or change_score >= self.frame_change_threshold
            event = _event(
                len(events),
                _elapsed_ms(started),
                state,
                _format_action(action_name, payload),
                before,
            )
            event.after_frame = _gray(after).copy()
            event.frame_change_score = change_score
            event.input_backend = input_backend
            event.error = error
            events.append(event)
            current_frame = after
            if error:
                return StartupResult(
                    success=False,
                    state=last_state,
                    elapsed_ms=_elapsed_ms(started),
                    attempts=actions,
                    reason="MENU_INPUT_FAILED",
                    events=events,
                )

            layout, current_frame, post_action_state = self._probe_after_start_action(
                deadline,
                current_frame,
            )
            last_state = post_action_state
            if layout is not None:
                elapsed = _elapsed_ms(started)
                events.append(
                    _event(len(events), elapsed, StartupState.PLAYABLE, "resolved", current_frame)
                )
                return StartupResult(
                    success=True,
                    state=StartupState.PLAYABLE,
                    elapsed_ms=elapsed,
                    attempts=actions,
                    layout=layout,
                    events=events,
                )
            if _state_accepts_startup_input(post_action_state):
                continue
            break

        if actions > 0 and not frame_changed:
            last_reason = "MENU_INPUT_FAILED"
        return StartupResult(
            success=False,
            state=last_state,
            elapsed_ms=_elapsed_ms(started),
            attempts=actions,
            reason=last_reason,
            events=events,
        )

    def _startup_actions(self) -> list[tuple[str, tuple[int, int] | str]]:
        candidates: list[tuple[str, tuple[int, int] | str]] = [
            ("click_primary", self.title_click),
        ]
        if self.startup_key is not None:
            candidates.append(("key", self.startup_key))
        candidates.append(("click_fallback", self.confirm_click))
        return [candidates[index % len(candidates)] for index in range(max(0, self.max_actions))]

    def _wait_for_actionable_frame(self, deadline: float) -> tuple[np.ndarray | None, StartupState]:
        last_state = StartupState.DARK_LOADING
        while time.monotonic() < deadline:
            frame = self.full_frame_getter()
            state = classify_startup_frame(frame, self.references)
            last_state = state
            if state == StartupState.PLAYABLE or _state_accepts_startup_input(state):
                return frame, state
            time.sleep(0.1)
        return None, last_state

    def _probe_until(self, deadline: float) -> Any | None:
        while time.monotonic() < deadline:
            try:
                return self.resolve_layout()
            except Exception:
                time.sleep(self.probe_interval)
        return None

    def _probe_after_start_action(
        self,
        deadline: float,
        current_frame: np.ndarray,
    ) -> tuple[Any | None, np.ndarray, StartupState]:
        state = classify_startup_frame(current_frame, self.references)
        first_probe = True
        while time.monotonic() < deadline:
            if not first_probe and _state_accepts_startup_input(state):
                return None, current_frame, state
            first_probe = False
            layout = self._probe_until(
                min(deadline, time.monotonic() + max(0.05, self.probe_interval))
            )
            if layout is not None:
                return layout, current_frame, StartupState.PLAYABLE
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.1, self.probe_interval))
            current_frame = self.full_frame_getter()
            state = classify_startup_frame(current_frame, self.references)
        return None, current_frame, state


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
    if _title_menu_overlay_visible(gray):
        return StartupState.TITLE_MENU
    if std > 18.0:
        return StartupState.TITLE_BACKGROUND
    return StartupState.UNKNOWN


def load_reference_frames(reference_dir: str | Path | None) -> dict[StartupState, np.ndarray]:
    if reference_dir is None:
        return {}
    base = Path(reference_dir).expanduser()
    if not base.exists():
        return {}
    mapping = {
        StartupState.DARK_LOADING: "dark_loading.png",
        StartupState.TITLE_BACKGROUND: "title_background.png",
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


def _frame_change_score(before: np.ndarray, after: np.ndarray) -> float:
    before_gray = _gray(before)
    after_gray = _gray(after)
    if before_gray.shape != after_gray.shape:
        after_gray = _resize_nearest(after_gray, before_gray.shape[0], before_gray.shape[1])
    return float(np.mean(np.abs(before_gray.astype(np.int16) - after_gray.astype(np.int16))))


def _state_accepts_startup_input(state: StartupState) -> bool:
    return state in {
        StartupState.TITLE_MENU,
        StartupState.CONFIRM_DIALOG,
        StartupState.SETTINGS_MENU,
    }


def _title_menu_overlay_visible(gray: np.ndarray) -> bool:
    if gray.ndim != 2:
        gray = _gray(gray)
    height, width = gray.shape
    if height < 64 or width < 64:
        return False
    title_roi = _relative_crop(gray, x0=0.18, y0=0.10, x1=0.50, y1=0.27)
    menu_roi = _relative_crop(gray, x0=0.54, y0=0.22, x1=0.84, y1=0.70)
    title_edge = _edge_density(title_roi, threshold=25)
    title_bright = _relative_bright_ratio(title_roi, margin=35)
    menu_edge = _edge_density(menu_roi, threshold=25)
    menu_dark = _relative_dark_ratio(menu_roi, margin=18)
    return (
        title_edge >= 0.045
        and title_bright >= 0.12
        and menu_edge >= 0.022
        and menu_dark >= 0.10
    )


def _relative_crop(
    gray: np.ndarray,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    height, width = gray.shape
    left = max(0, min(width - 1, int(width * x0)))
    right = max(left + 1, min(width, int(width * x1)))
    top = max(0, min(height - 1, int(height * y0)))
    bottom = max(top + 1, min(height, int(height * y1)))
    return gray[top:bottom, left:right]


def _edge_density(gray: np.ndarray, *, threshold: int) -> float:
    if gray.size == 0:
        return 0.0
    values = gray.astype(np.int16, copy=False)
    density = 0.0
    if values.shape[1] > 1:
        density += float((np.abs(np.diff(values, axis=1)) > threshold).mean())
    if values.shape[0] > 1:
        density += float((np.abs(np.diff(values, axis=0)) > threshold).mean())
    return density


def _relative_bright_ratio(gray: np.ndarray, *, margin: int) -> float:
    if gray.size == 0:
        return 0.0
    median = float(np.median(gray))
    return float((gray.astype(np.int16, copy=False) > median + margin).mean())


def _relative_dark_ratio(gray: np.ndarray, *, margin: int) -> float:
    if gray.size == 0:
        return 0.0
    median = float(np.median(gray))
    return float((gray.astype(np.int16, copy=False) < median - margin).mean())


def _format_action(action_name: str, payload: tuple[int, int] | str) -> str:
    if isinstance(payload, tuple):
        return f"{action_name}:{payload[0]},{payload[1]}"
    return f"{action_name}:{payload}"


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
