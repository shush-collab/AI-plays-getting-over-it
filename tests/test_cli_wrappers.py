import json
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


class CliWrapperTests(unittest.TestCase):
    def test_observation_schema_wrapper_emits_json(self) -> None:
        result = run_script("goi_observation_schema.py", "--body-rays", "4", "--hammer-rays", "4")
        payload = json.loads(result.stdout)

        self.assertEqual(payload["version"], "v1")
        self.assertEqual(payload["body_ray_count"], 4)
        self.assertEqual(payload["hammer_ray_count"], 4)

    def test_game_facing_wrappers_expose_help(self) -> None:
        for script_name in (
            "goi_live_position.py",
            "goi_memory_probe.py",
            "goi_observation_state.py",
            "goi_ptrace_il2cpp.py",
        ):
            with self.subTest(script_name=script_name):
                result = run_script(script_name, "--help")
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
