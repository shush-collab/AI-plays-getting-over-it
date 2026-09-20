import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.game.runtime.window_control import WindowControl  # noqa: E402


class WindowControlTests(unittest.TestCase):
    def test_xdotool_absolute_click_uses_screen_coordinates(self) -> None:
        calls = []

        def fake_run(command, *, timeout):
            calls.append(command)

        control = WindowControl(input_backend="xdotool")
        with patch("aiget.game.runtime.window_control._run", side_effect=fake_run):
            control.click_absolute(1595, 483)

        self.assertEqual(calls, [["xdotool", "mousemove", "1595", "483", "click", "1"]])

    def test_auto_wayland_falls_back_to_xdotool_when_ydotool_missing(self) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("XDG_SESSION_TYPE", "WAYLAND_DISPLAY")
        }
        env["XDG_SESSION_TYPE"] = "wayland"
        calls = []
        control = WindowControl(input_backend="auto")

        def fake_which(name):
            return "/usr/bin/xdotool" if name == "xdotool" else None

        def fake_run(command, *, timeout):
            calls.append(command)

        with (
            patch.dict(os.environ, env, clear=True),
            patch("aiget.game.runtime.window_control.shutil.which", side_effect=fake_which),
            patch("aiget.game.runtime.window_control._run", side_effect=fake_run),
        ):
            control.click_absolute(1, 2)

        self.assertEqual(calls, [["xdotool", "mousemove", "1", "2", "click", "1"]])
        self.assertEqual(control.active_input_backend, "xdotool")

    def test_auto_wayland_prefers_ydotool_when_available(self) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("XDG_SESSION_TYPE", "WAYLAND_DISPLAY")
        }
        env["XDG_SESSION_TYPE"] = "wayland"
        calls = []
        control = WindowControl(input_backend="auto")

        def fake_which(name):
            return f"/usr/bin/{name}"

        def fake_run(command, *, timeout):
            calls.append(command)

        with (
            patch.dict(os.environ, env, clear=True),
            patch("aiget.game.runtime.window_control.shutil.which", side_effect=fake_which),
            patch("aiget.game.runtime.window_control._run", side_effect=fake_run),
        ):
            control.click_absolute(1, 2)

        self.assertEqual(
            calls,
            [
                ["ydotool", "mousemove", "--absolute", "1", "2"],
                ["ydotool", "click", "0xC0"],
            ],
        )
        self.assertEqual(control.active_input_backend, "ydotool")

    def test_auto_requires_some_input_backend(self) -> None:
        control = WindowControl(input_backend="auto")

        with patch("aiget.game.runtime.window_control.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "xdotool"):
                control.click_absolute(1, 2)


if __name__ == "__main__":
    unittest.main()
