import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.reset_startup import (  # noqa: E402
    StartupAutomation,
    StartupState,
    classify_startup_frame,
)


class DummyWindowControl:
    def __init__(self):
        self.clicks = []
        self.keys = []

    def click_relative(self, x: int, y: int) -> None:
        self.clicks.append((x, y))

    def key(self, key: str) -> None:
        self.keys.append(key)


class ResetStartupTests(unittest.TestCase):
    def test_dark_frame_classifies_as_loading(self) -> None:
        frame = np.zeros((84, 84, 4), dtype=np.uint8)

        self.assertEqual(classify_startup_frame(frame), StartupState.DARK_LOADING)

    def test_references_override_heuristics(self) -> None:
        frame = np.indices((84, 84)).sum(axis=0).astype(np.uint8)
        refs = {StartupState.SETTINGS_MENU: frame.copy()}

        self.assertEqual(classify_startup_frame(frame, refs), StartupState.SETTINGS_MENU)

    def test_automation_returns_layout_when_resolver_succeeds(self) -> None:
        control = DummyWindowControl()
        expected_layout = object()
        automation = StartupAutomation(
            frame_getter=_nonblank_frame,
            window_control=control,
            resolve_layout=lambda: expected_layout,
            timeout=1.0,
        )

        result = automation.drive_until_playable()

        self.assertTrue(result.success)
        self.assertIs(result.layout, expected_layout)
        self.assertEqual(result.state, StartupState.PLAYABLE)
        self.assertEqual(control.clicks, [])

    def test_automation_uses_primary_then_fallback_clicks(self) -> None:
        control = DummyWindowControl()
        calls = {"count": 0}

        def resolve_layout():
            calls["count"] += 1
            raise RuntimeError("FindObjectOfType(PlayerControl) returned null")

        automation = StartupAutomation(
            frame_getter=_nonblank_frame,
            full_frame_getter=_nonblank_full_frame,
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


def _nonblank_frame() -> np.ndarray:
    gray = _nonblank_full_frame()[::9, ::16]
    return np.repeat(gray[:, :, None], 4, axis=2)


def _nonblank_full_frame() -> np.ndarray:
    y, x = np.indices((720, 1280))
    return ((x + y) % 255).astype(np.uint8)


if __name__ == "__main__":
    unittest.main()
