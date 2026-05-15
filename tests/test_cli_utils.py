import argparse
import unittest

from aiget.cli_utils import parse_args_allowing_launch_flags


class CliUtilsTests(unittest.TestCase):
    def test_launch_command_accepts_steam_flags_before_other_args(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--launch-command", nargs="+", default=None)
        parser.add_argument("--clean-save-path")

        args = parse_args_allowing_launch_flags(
            parser,
            [
                "--launch-command",
                "steam",
                "-applaunch",
                "240720",
                "--",
                "-screen-width",
                "1920",
                "--clean-save-path",
                "/tmp/clean",
            ],
        )

        self.assertEqual(
            args.launch_command,
            ["steam", "-applaunch", "240720", "--", "-screen-width", "1920"],
        )
        self.assertEqual(args.clean_save_path, "/tmp/clean")


if __name__ == "__main__":
    unittest.main()
