import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.live_position import CandidatePath, fallback_candidate_from_truth


class _FakeReader:
    def __init__(self, vectors: dict[int, tuple[float, float]]):
        self.vectors = vectors

    def read_vec2(self, addr: int) -> tuple[float, float]:
        return self.vectors[addr]


class LivePositionTests(unittest.TestCase):
    def test_fallback_candidate_uses_known_rb_native_path(self) -> None:
        roots = {"rb_native": 0x1000, "tf_native": 0x2000}
        reader = _FakeReader({0x10A8: (13.2, 63.35)})

        matches = fallback_candidate_from_truth(
            reader,
            roots=roots,
            target_x=13.185,
            target_y=63.352,
            eps=0.0015,
        )

        self.assertEqual(matches, {CandidatePath(root_name="rb_native", ptr_offset=None, value_offset=0xA8)})

    def test_fallback_candidate_rejects_distant_values(self) -> None:
        roots = {"rb_native": 0x1000, "tf_native": 0x2000}
        reader = _FakeReader({0x10A8: (10.0, 50.0)})

        matches = fallback_candidate_from_truth(
            reader,
            roots=roots,
            target_x=13.185,
            target_y=63.352,
            eps=0.0015,
        )

        self.assertEqual(matches, set())


if __name__ == "__main__":
    unittest.main()
