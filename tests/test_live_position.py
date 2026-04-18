import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.live_position import CandidatePath, FastCursorLane, fallback_candidate_from_truth, read_fast_cursor_sample


class _FakeReader:
    def __init__(self, vectors: dict[int, tuple[float, float]], ptrs: dict[int, int] | None = None):
        self.vectors = vectors
        self.ptrs = ptrs or {}

    def read_vec2(self, addr: int) -> tuple[float, float]:
        return self.vectors[addr]

    def read_ptr(self, addr: int) -> int:
        return self.ptrs[addr]


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

    def test_read_fast_cursor_sample_uses_frozen_lane_without_indirection(self) -> None:
        reader = _FakeReader({0x10A8: (13.2, 63.35)})
        lane = FastCursorLane(
            pid=1234,
            fake_cursor_native=0x2000,
            fake_cursor_rb_native=0x1000,
            candidate=CandidatePath(root_name="rb_native", ptr_offset=None, value_offset=0xA8),
            current_addr=0x10A8,
            calibration_samples=1,
            calibration_hits=1,
            motion_span=0.0,
        )

        sample = read_fast_cursor_sample(reader, lane, ts=42.0)

        self.assertEqual(sample.ts, 42.0)
        self.assertEqual(sample.pid, 1234)
        self.assertEqual(sample.addr, 0x10A8)
        self.assertEqual((sample.x, sample.y), (13.2, 63.35))

    def test_read_fast_cursor_sample_supports_indirection(self) -> None:
        reader = _FakeReader({0x30A8: (7.5, 8.5)}, ptrs={0x2010: 0x3000})
        lane = FastCursorLane(
            pid=4321,
            fake_cursor_native=0x2000,
            fake_cursor_rb_native=0x1000,
            candidate=CandidatePath(root_name="tf_native", ptr_offset=0x10, value_offset=0xA8),
            current_addr=0x30A8,
            calibration_samples=1,
            calibration_hits=1,
            motion_span=0.0,
        )

        sample = read_fast_cursor_sample(reader, lane, ts=7.0)

        self.assertEqual(sample.addr, 0x30A8)
        self.assertEqual((sample.x, sample.y), (7.5, 8.5))


if __name__ == "__main__":
    unittest.main()
