import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiget.game.runtime.cli_utils import parse_args_allowing_launch_flags
from aiget.game.runtime.frame_capture import CaptureRegion
from aiget.game.runtime.reset_test import (
    ResetConfig,
    build_parser,
    config_from_args,
    failure_stage,
    game_modules_ready,
    restore_save,
    validate_playable_frame,
)


def make_config() -> ResetConfig:
    return ResetConfig(
        launch_command=("steam",),
        clean_save_path=Path("clean"),
        active_save_path=Path("active"),
        capture_region=CaptureRegion(left=0, top=0, width=4, height=4),
        game_ready_timeout=1.0,
        image_std_threshold=10.0,
        image_mean_min=5.0,
        image_mean_max=250.0,
        startup_input_backend="auto",
        startup_reference_dir=None,
        startup_title_click=(1, 1),
        startup_confirm_click=(2, 2),
        startup_key=None,
        startup_attempts=1,
        startup_probe_timeout=1.0,
        window_title="Getting Over It",
        window_left=None,
        window_top=None,
        window_width=4,
        window_height=4,
        kill_existing=True,
        calibration_samples=1,
        calibration_interval=0.1,
        layout_window=2,
        layout_eps=0.001,
    )


class ResetTestTests(unittest.TestCase):
    def test_restore_save_replaces_an_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clean = root / "clean"
            active = root / "active"
            clean.mkdir()
            (clean / "state.txt").write_text("clean", encoding="utf-8")
            active.mkdir()
            (active / "state.txt").write_text("dirty", encoding="utf-8")

            restore_save(clean, active)

            self.assertEqual((active / "state.txt").read_text(encoding="utf-8"), "clean")

    def test_game_modules_ready_requires_both_unity_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            maps = Path(tmpdir) / "123" / "maps"
            maps.parent.mkdir()
            maps.write_text("GameAssembly.so\nUnityPlayer.so\n", encoding="utf-8")

            self.assertTrue(game_modules_ready(123, Path(tmpdir)))
            maps.write_text("GameAssembly.so\n", encoding="utf-8")
            self.assertFalse(game_modules_ready(123, Path(tmpdir)))

    def test_playable_frame_rejects_blank_images(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "image_std"):
            validate_playable_frame(np.zeros((4, 4), dtype=np.uint8), make_config())

    def test_playable_frame_accepts_varied_gray_image(self) -> None:
        frame = np.array([[10, 30], [80, 150]], dtype=np.uint8)
        validate_playable_frame(frame, make_config())

    def test_launch_command_accepts_steam_flags(self) -> None:
        args = parse_args_allowing_launch_flags(
            build_parser(),
            [
                "--launch-command",
                "steam",
                "-applaunch",
                "240720",
                "--clean-save-path",
                "clean",
                "--active-save-path",
                "active",
                "--capture-left",
                "0",
                "--capture-top",
                "0",
                "--capture-width",
                "4",
                "--capture-height",
                "4",
            ],
        )
        config = config_from_args(args)

        self.assertEqual(config.launch_command, ("steam", "-applaunch", "240720"))

    def test_failure_stage_keeps_known_startup_reason(self) -> None:
        self.assertEqual(
            failure_stage(RuntimeError("MENU_VISIBLE_TIMEOUT")),
            "MENU_VISIBLE_TIMEOUT",
        )


if __name__ == "__main__":
    unittest.main()
