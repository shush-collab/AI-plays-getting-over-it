import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.png_utils import gray_to_rgb, read_png, write_gray_png, write_rgb_png  # noqa: E402


class PngUtilsTests(unittest.TestCase):
    def test_gray_png_round_trip(self) -> None:
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gray.png"
            write_gray_png(path, image)

            decoded = read_png(path)

        np.testing.assert_array_equal(decoded, image)

    def test_rgb_png_round_trip(self) -> None:
        image = gray_to_rgb(np.arange(12, dtype=np.uint8).reshape(3, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.png"
            write_rgb_png(path, image)

            decoded = read_png(path)

        np.testing.assert_array_equal(decoded, image)


if __name__ == "__main__":
    unittest.main()
