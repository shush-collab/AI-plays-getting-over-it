import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    def test_contributing_guide_exists(self) -> None:
        self.assertTrue((PROJECT_ROOT / "contributions.md").is_file())

    def test_source_package_exists(self) -> None:
        self.assertTrue((PROJECT_ROOT / "src" / "aiget" / "__init__.py").is_file())


if __name__ == "__main__":
    unittest.main()
