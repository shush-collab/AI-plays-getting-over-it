import sys
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
