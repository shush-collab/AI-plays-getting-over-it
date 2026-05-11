import sys
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.env import IMAGE_SHAPE, RESET_RELAUNCH, GettingOverItEnv  # noqa: E402


class EnvSafetyTests(unittest.TestCase):
    def test_image_observation_is_four_frame_stack(self) -> None:
        self.assertEqual(IMAGE_SHAPE, (84, 84, 4))
        self.assertEqual(GettingOverItEnv.observation_space["image"].shape, IMAGE_SHAPE)

    def test_relaunch_reset_requires_launch_command(self) -> None:
        env = GettingOverItEnv(
            reset_backend=RESET_RELAUNCH,
            clean_save_path="/tmp/aiget-clean-save",
            active_save_path="/tmp/aiget-active-save",
            enable_image=False,
            enable_uinput=False,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "launch_command"):
                env._reset_game_process()
        finally:
            env.close()

    def test_short_cursor_read_is_process_loss(self) -> None:
        env = GettingOverItEnv(enable_image=False, enable_uinput=False)
        env._mem_fd = 123
        env._fast_addr = 0x456
        try:
            with patch("aiget.env.os.pread", return_value=b"\x00"):
                with self.assertRaisesRegex(ProcessLookupError, "short cursor read"):
                    env.read_observation_vector()
        finally:
            env._mem_fd = -1
            env.close()

    def test_startup_click_uses_live_window_origin(self) -> None:
        env = GettingOverItEnv(
            enable_image=False,
            enable_uinput=False,
            startup_click=(10, 20),
            startup_attempts=1,
        )
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if "getwindowgeometry" in command:
                return CompletedProcess(command, 0, stdout="X=100\nY=200\n", stderr="")
            return CompletedProcess(command, 0, stdout="", stderr="")

        try:
            with patch("aiget.env.subprocess.run", side_effect=fake_run):
                env._send_startup_action()
        finally:
            env.close()

        self.assertIn(["xdotool", "mousemove", "110", "220", "click", "1"], calls)


if __name__ == "__main__":
    unittest.main()
