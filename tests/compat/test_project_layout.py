import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProjectLayoutTests(unittest.TestCase):
    def test_contributing_guide_exists(self) -> None:
        self.assertTrue((PROJECT_ROOT / "contributions.md").is_file())

    def test_source_package_exists(self) -> None:
        self.assertTrue((PROJECT_ROOT / "src" / "aiget" / "__init__.py").is_file())

    def test_domain_directories_exist(self) -> None:
        for path in (
            "src/aiget/game/probing",
            "src/aiget/game/runtime",
            "src/aiget/game/worker",
            "src/aiget/rl",
            "src/aiget/shared",
            "game/worker-plugin",
            "simulation",
            "docs/game",
            "docs/rl",
            "scripts/game",
        ):
            with self.subTest(path=path):
                self.assertTrue((PROJECT_ROOT / path).is_dir())

    def test_legacy_script_shims_exist(self) -> None:
        for path in ("scripts/build_physics_worker_plugin.sh", "scripts/reset_probe.sh"):
            with self.subTest(path=path):
                self.assertTrue((PROJECT_ROOT / path).is_file())


if __name__ == "__main__":
    unittest.main()
