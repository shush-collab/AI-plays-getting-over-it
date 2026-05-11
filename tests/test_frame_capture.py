import struct
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.frame_capture import _decode_xwd  # noqa: E402


class FrameCaptureTests(unittest.TestCase):
    def test_decode_xwd_bgra_payload(self) -> None:
        width = 2
        height = 2
        name = b"xwd\x00"
        header_size = 100 + len(name)
        bytes_per_line = width * 4
        header = struct.pack(
            ">25I",
            header_size,
            7,
            2,
            24,
            width,
            height,
            0,
            0,
            32,
            0,
            32,
            32,
            bytes_per_line,
            5,
            0xFF0000,
            0x00FF00,
            0x0000FF,
            8,
            0,
            0,
            width,
            height,
            0,
            0,
            0,
        )
        payload = bytes(
            [
                1,
                2,
                3,
                255,
                4,
                5,
                6,
                255,
                7,
                8,
                9,
                255,
                10,
                11,
                12,
                255,
            ]
        )

        frame = _decode_xwd(header + name + payload)

        self.assertEqual(frame.shape, (2, 2, 4))
        np.testing.assert_array_equal(frame[0, 0], np.array([1, 2, 3, 255], dtype=np.uint8))
        np.testing.assert_array_equal(frame[1, 1], np.array([10, 11, 12, 255], dtype=np.uint8))

    def test_decode_xwd_rejects_short_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "too short"):
            _decode_xwd(b"\x00")


if __name__ == "__main__":
    unittest.main()
