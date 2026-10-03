import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.game.runtime.reset_startup import (  # noqa: E402
    StartupAutomation,
    StartupState,
    classify_startup_frame,
)
from aiget.shared.png_utils import write_gray_png  # noqa: E402


class DummyWindowControl:
    def __init__(self):
        self.clicks = []
        self.keys = []

    def click_relative(self, x: int, y: int) -> None:
        self.clicks.append((x, y))

    def key(self, key: str) -> None:
        self.keys.append(key)

    @property
    def active_input_backend(self) -> str:
        return "dummy"


class ResetStartupTests(unittest.TestCase):
    def test_dark_frame_classifies_as_loading(self) -> None:
        frame = np.zeros((84, 84, 4), dtype=np.uint8)

        self.assertEqual(classify_startup_frame(frame), StartupState.DARK_LOADING)

    def test_references_override_heuristics(self) -> None:
        frame = np.indices((84, 84)).sum(axis=0).astype(np.uint8)
        refs = {StartupState.SETTINGS_MENU: frame.copy()}

        self.assertEqual(classify_startup_frame(frame, refs), StartupState.SETTINGS_MENU)

    def test_title_background_does_not_classify_as_menu(self) -> None:
        self.assertEqual(
            classify_startup_frame(_title_background_frame()),
            StartupState.TITLE_BACKGROUND,
        )

    def test_title_menu_requires_visible_overlay(self) -> None:
        self.assertEqual(classify_startup_frame(_title_menu_frame()), StartupState.TITLE_MENU)

    def test_automation_returns_layout_when_resolver_succeeds(self) -> None:
        control = DummyWindowControl()
        expected_layout = object()
        with tempfile.TemporaryDirectory() as tmp:
            write_gray_png(Path(tmp) / "playable.png", _nonblank_full_frame())
            automation = StartupAutomation(
                frame_getter=_nonblank_frame,
                full_frame_getter=_nonblank_full_frame,
                window_control=control,
                resolve_layout=lambda: expected_layout,
                reference_dir=tmp,
                timeout=1.0,
            )

            result = automation.drive_until_playable()

        self.assertTrue(result.success)
        self.assertIs(result.layout, expected_layout)
        self.assertEqual(result.state, StartupState.PLAYABLE)
        self.assertEqual(control.clicks, [])

    def test_automation_clicks_before_first_title_menu_probe(self) -> None:
        events = []

        class OrderedControl(DummyWindowControl):
            def click_relative(self, x: int, y: int) -> None:
                events.append("click")
                super().click_relative(x, y)

        control = OrderedControl()
        expected_layout = object()

        def resolve_layout():
            events.append("probe")
            return expected_layout

        automation = StartupAutomation(
            frame_getter=_title_menu_small_frame,
            full_frame_getter=_title_menu_frame,
            window_control=control,
            resolve_layout=resolve_layout,
            timeout=1.0,
            probe_interval=0.05,
            action_interval=0.1,
            title_click=(10, 20),
        )

        result = automation.drive_until_playable()

        self.assertTrue(result.success)
        self.assertIs(result.layout, expected_layout)
        self.assertEqual(events[:2], ["click", "probe"])
        self.assertEqual(control.clicks, [(10, 20)])

    def test_automation_waits_for_menu_overlay_before_clicking(self) -> None:
        control = DummyWindowControl()
        frames = [_title_background_frame(), _title_background_frame(), _title_menu_frame()]
        calls = {"frames": 0}
        expected_layout = object()

        def full_frame_getter():
            calls["frames"] += 1
            if frames:
                return frames.pop(0)
            return _title_menu_frame()

        automation = StartupAutomation(
            frame_getter=_title_menu_small_frame,
            full_frame_getter=full_frame_getter,
            window_control=control,
            resolve_layout=lambda: expected_layout,
            timeout=1.0,
            probe_interval=0.01,
            action_interval=0.01,
            title_click=(10, 20),
        )

        result = automation.drive_until_playable()

        self.assertTrue(result.success)
        self.assertEqual(control.clicks, [(10, 20)])
        self.assertGreaterEqual(calls["frames"], 3)

    def test_automation_keeps_probing_after_click_transition(self) -> None:
        control = DummyWindowControl()
        frames = [_title_menu_frame(), _gameplay_frame(), _gameplay_frame()]
        calls = {"probes": 0}
        expected_layout = object()

        def full_frame_getter():
            if frames:
                return frames.pop(0)
            return _gameplay_frame()

        def resolve_layout():
            calls["probes"] += 1
            if calls["probes"] < 2:
                raise RuntimeError("FindObjectOfType(PlayerControl) returned null")
            return expected_layout

        automation = StartupAutomation(
            frame_getter=_title_menu_small_frame,
            full_frame_getter=full_frame_getter,
            window_control=control,
            resolve_layout=resolve_layout,
            timeout=0.8,
            probe_interval=0.01,
            action_interval=0.01,
            title_click=(10, 20),
        )

        result = automation.drive_until_playable()

        self.assertTrue(result.success)
        self.assertIs(result.layout, expected_layout)
        self.assertEqual(control.clicks, [(10, 20)])
        self.assertGreaterEqual(calls["probes"], 2)

    def test_automation_reports_menu_visible_timeout_state(self) -> None:
        control = DummyWindowControl()
        automation = StartupAutomation(
            frame_getter=_title_menu_small_frame,
            full_frame_getter=_title_background_frame,
            window_control=control,
            resolve_layout=lambda: object(),
            timeout=0.15,
            probe_interval=0.01,
            action_interval=0.01,
        )

        result = automation.drive_until_playable()

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "MENU_VISIBLE_TIMEOUT")
        self.assertEqual(result.state, StartupState.TITLE_BACKGROUND)
        self.assertEqual(control.clicks, [])

    def test_automation_uses_primary_then_fallback_clicks(self) -> None:
        control = DummyWindowControl()
        calls = {"count": 0}

        def resolve_layout():
            calls["count"] += 1
            raise RuntimeError("FindObjectOfType(PlayerControl) returned null")

        automation = StartupAutomation(
            frame_getter=_title_menu_small_frame,
            full_frame_getter=_title_menu_frame,
            window_control=control,
            resolve_layout=resolve_layout,
            timeout=0.8,
            probe_interval=0.05,
            action_interval=0.1,
            title_click=(10, 20),
            confirm_click=(30, 40),
        )

        result = automation.drive_until_playable()

        self.assertFalse(result.success)
        self.assertEqual(control.clicks, [(10, 20), (30, 40)])
        self.assertEqual(result.reason, "MENU_INPUT_FAILED")


def _nonblank_frame() -> np.ndarray:
    gray = _nonblank_full_frame()[::9, ::16]
    return np.repeat(gray[:, :, None], 4, axis=2)


def _nonblank_full_frame() -> np.ndarray:
    y, x = np.indices((720, 1280))
    return ((x + y) % 255).astype(np.uint8)


def _title_background_frame() -> np.ndarray:
    height, width = 720, 1280
    y, x = np.indices((height, width))
    frame = np.full((height, width), 165, dtype=np.uint8)
    rock = x < (width * 0.48 + 40 * np.sin(y / 90.0))
    frame[rock] = np.clip(42 + (x[rock] // 20) + (y[rock] // 25), 0, 115).astype(np.uint8)
    handle = (x > int(width * 0.53)) & (x < int(width * 0.58)) & (y > int(height * 0.18))
    frame[handle] = 95
    hammer = (x > int(width * 0.46)) & (x < int(width * 0.68)) & (y > 88) & (y < 150)
    frame[hammer] = 120
    frame[16:45, 1120:1240] = 205
    return frame


def _title_menu_frame() -> np.ndarray:
    frame = _title_background_frame()
    _draw_block_text(frame, x=290, y=95, width=295, height=42, strokes=9, value=235)
    _draw_block_text(frame, x=305, y=145, width=280, height=30, strokes=8, value=235)
    for index, width in enumerate((190, 205, 160, 145, 85)):
        _draw_block_text(
            frame,
            x=770,
            y=190 + index * 62,
            width=width,
            height=35,
            strokes=7,
            value=105,
        )
    return frame


def _title_menu_small_frame() -> np.ndarray:
    gray = _title_menu_frame()[::9, ::16]
    return np.repeat(gray[:, :, None], 4, axis=2)


def _gameplay_frame() -> np.ndarray:
    height, width = 720, 1280
    y, x = np.indices((height, width))
    frame = np.full((height, width), 62, dtype=np.uint8)
    ground = y > (height * 0.74 + 15 * np.sin(x / 80.0))
    frame[ground] = 205
    body = (x - 500) ** 2 + (y - 430) ** 2 < 42**2
    frame[body] = 35
    hammer = (x > 470) & (x < 535) & (y > 220) & (y < 420)
    frame[hammer] = 180
    rocks = (x > 710) & (y > 330) & (y < 555)
    frame[rocks] = 115
    return frame


def _draw_block_text(
    frame: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    strokes: int,
    value: int,
) -> None:
    for offset in range(strokes):
        left = x + offset * max(8, width // (strokes + 1))
        right = min(frame.shape[1], left + max(4, width // 30))
        frame[y : y + height, left:right] = value
    frame[y : y + max(4, height // 6), x : x + width] = value
    frame[y + height - max(4, height // 6) : y + height, x : x + width] = value


if __name__ == "__main__":
    unittest.main()
