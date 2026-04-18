import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.observation_state import PositionSample, ProgressTracker, estimate_velocity, format_payload


class ObservationStateTests(unittest.TestCase):
    def test_estimate_velocity_from_two_samples(self) -> None:
        previous = PositionSample(ts=10.0, x=1.0, y=3.0)
        current = PositionSample(ts=10.5, x=2.5, y=2.0)

        velocity = estimate_velocity(previous, current)

        self.assertEqual(velocity, (3.0, -2.0))

    def test_estimate_velocity_handles_missing_or_invalid_dt(self) -> None:
        current = PositionSample(ts=10.0, x=2.0, y=4.0)

        self.assertEqual(estimate_velocity(None, current), (0.0, 0.0))
        self.assertEqual(
            estimate_velocity(PositionSample(ts=10.0, x=1.0, y=1.0), current),
            (0.0, 0.0),
        )

    def test_progress_tracker_updates_best_height_and_timer(self) -> None:
        tracker = ProgressTracker(best_height=5.0, last_progress_ts=100.0)

        stalled = tracker.update(current_height=5.0, now=101.5)
        improved = tracker.update(current_height=5.5, now=103.0)

        self.assertEqual(stalled, (5.0, 5.0, 1.5))
        self.assertEqual(improved, (5.5, 5.5, 0.0))

    def test_format_payload_exposes_implemented_features(self) -> None:
        payload = format_payload(
            ts=123.0,
            pid=456,
            addr=0xABC,
            x=1.0,
            y=2.0,
            vx=3.0,
            vy=4.0,
            progress=(2.0, 5.0, 7.5),
        )

        self.assertEqual(payload["implemented_features"], ["cursor_position_xy", "cursor_velocity_xy", "progress_features"])
        self.assertEqual(payload["cursor_position_xy"], [1.0, 2.0])
        self.assertEqual(payload["cursor_velocity_xy"], [3.0, 4.0])
        self.assertEqual(payload["progress_features"], [2.0, 5.0, 7.5])


if __name__ == "__main__":
    unittest.main()
