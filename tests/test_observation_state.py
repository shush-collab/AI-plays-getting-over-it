import math
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.observation_state import (
    AngleSample,
    ObservationRefs,
    ObservationSnapshot,
    PositionSample,
    ProgressTracker,
    RichStateSnapshot,
    SlowLaneAccumulator,
    SlowObservationLane,
    SlowObservationSample,
    build_rich_state_snapshot,
    consume_latest_rich_state,
    decode_hammer_contact_state,
    estimate_angular_velocity,
    estimate_velocity,
    format_payload,
    freeze_slow_observation_lane,
    publish_latest_rich_state,
    quaternion_to_z_angle,
    read_slow_observation_sample,
)


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

    def test_estimate_angular_velocity_wraps_at_pi_boundary(self) -> None:
        previous = AngleSample(ts=5.0, angle=math.pi - 0.1)
        current = AngleSample(ts=5.5, angle=-math.pi + 0.1)

        angular_velocity = estimate_angular_velocity(previous, current)

        self.assertAlmostEqual(angular_velocity, 0.4)

    def test_progress_tracker_updates_best_height_and_timer(self) -> None:
        tracker = ProgressTracker(best_height=5.0, last_progress_ts=100.0)

        stalled = tracker.update(current_height=5.0, now=101.5)
        improved = tracker.update(current_height=5.5, now=103.0)

        self.assertEqual(stalled, (5.0, 5.0, 1.5))
        self.assertEqual(improved, (5.5, 5.5, 0.0))

    def test_freeze_slow_observation_lane_keeps_refs_and_refresh_interval(self) -> None:
        refs = ObservationRefs(
            player_obj=1,
            fake_cursor_obj=2,
            fake_cursor_native=3,
            fake_cursor_rb_obj=4,
            fake_cursor_rb_native=5,
            body_transform_obj=6,
            hammer_anchor_transform_obj=7,
            hammer_tip_transform_obj=8,
            hammer_collisions_obj=9,
        )

        with patch("aiget.observation_state.resolve_observation_refs", return_value=refs):
            lane = freeze_slow_observation_lane(pid=1234, refresh_interval=0.75)

        self.assertEqual(lane.pid, 1234)
        self.assertEqual(lane.refs, refs)
        self.assertEqual(lane.refresh_interval, 0.75)

    def test_read_slow_observation_sample_wraps_snapshot_with_timestamp(self) -> None:
        lane = SlowObservationLane(
            pid=1234,
            refs=ObservationRefs(
                player_obj=1,
                fake_cursor_obj=2,
                fake_cursor_native=3,
                fake_cursor_rb_obj=4,
                fake_cursor_rb_native=5,
                body_transform_obj=6,
                hammer_anchor_transform_obj=7,
                hammer_tip_transform_obj=8,
                hammer_collisions_obj=9,
            ),
            refresh_interval=0.5,
        )
        snapshot = ObservationSnapshot(
            body_position_xy=(1.0, 2.0),
            body_angle=0.25,
            hammer_anchor_xy=(3.0, 4.0),
            hammer_tip_xy=(5.0, 6.0),
            hammer_contact_flags=(1.0, 0.0),
            hammer_contact_normal_xy=(-0.5, 0.75),
        )

        with patch("aiget.observation_state.sample_observation_snapshot", return_value=snapshot):
            sample = read_slow_observation_sample(lane, ts=99.0)

        self.assertEqual(sample.ts, 99.0)
        self.assertEqual(sample.snapshot, snapshot)

    def test_build_rich_state_snapshot_batches_slow_lane_fields(self) -> None:
        accumulator = SlowLaneAccumulator(
            tracker=ProgressTracker(best_height=5.0, last_progress_ts=100.0),
            previous_body=PositionSample(ts=100.0, x=4.0, y=5.0),
            previous_body_angle=AngleSample(ts=100.0, angle=0.0),
            previous_hammer_angle=AngleSample(ts=100.0, angle=0.0),
        )
        sample = SlowObservationSample(
            ts=101.0,
            snapshot=ObservationSnapshot(
                body_position_xy=(6.0, 7.0),
                body_angle=math.pi / 4.0,
                hammer_anchor_xy=(2.0, 3.0),
                hammer_tip_xy=(4.0, 6.0),
                hammer_contact_flags=(1.0, 0.0),
                hammer_contact_normal_xy=(-0.5, 0.75),
            ),
        )

        rich_snapshot = build_rich_state_snapshot(sample, accumulator)

        self.assertEqual(rich_snapshot.ts, 101.0)
        self.assertEqual(rich_snapshot.body_position_xy, (6.0, 7.0))
        self.assertEqual(rich_snapshot.body_velocity_xy, (2.0, 2.0))
        self.assertAlmostEqual(rich_snapshot.body_rotation_sin_cos[0], math.sin(math.pi / 4.0))
        self.assertAlmostEqual(rich_snapshot.body_rotation_sin_cos[1], math.cos(math.pi / 4.0))
        self.assertEqual(rich_snapshot.hammer_anchor_xy, (2.0, 3.0))
        self.assertEqual(rich_snapshot.hammer_tip_xy, (4.0, 6.0))
        self.assertEqual(rich_snapshot.hammer_contact_flags, (1.0, 0.0))
        self.assertEqual(rich_snapshot.progress_features, (7.0, 7.0, 0.0))
        self.assertTrue(rich_snapshot.valid)
        self.assertEqual(accumulator.previous_body, PositionSample(ts=101.0, x=6.0, y=7.0))
        self.assertEqual(accumulator.previous_body_angle, AngleSample(ts=101.0, angle=math.pi / 4.0))
        self.assertAlmostEqual(accumulator.previous_hammer_angle.angle, math.atan2(3.0, 2.0))

    def test_publish_and_consume_latest_rich_state_reuses_newest_snapshot(self) -> None:
        import queue

        updates: queue.Queue[RichStateSnapshot] = queue.Queue(maxsize=1)
        older = RichStateSnapshot(
            ts=10.0,
            body_position_xy=(1.0, 2.0),
            body_velocity_xy=(0.0, 0.0),
            body_rotation_sin_cos=(0.0, 1.0),
            body_angular_velocity=0.0,
            hammer_anchor_xy=(3.0, 4.0),
            hammer_tip_xy=(5.0, 6.0),
            hammer_direction_sin_cos=(0.0, 1.0),
            hammer_angular_velocity=0.0,
            hammer_contact_flags=(0.0, 0.0),
            hammer_contact_normal_xy=(0.0, 0.0),
            progress_features=(1.0, 1.0, 1.0),
        )
        newer = RichStateSnapshot(
            ts=11.0,
            body_position_xy=(7.0, 8.0),
            body_velocity_xy=(1.0, 2.0),
            body_rotation_sin_cos=(1.0, 0.0),
            body_angular_velocity=3.0,
            hammer_anchor_xy=(9.0, 10.0),
            hammer_tip_xy=(11.0, 12.0),
            hammer_direction_sin_cos=(1.0, 0.0),
            hammer_angular_velocity=4.0,
            hammer_contact_flags=(1.0, 1.0),
            hammer_contact_normal_xy=(-0.5, 0.75),
            progress_features=(2.0, 3.0, 4.0),
        )

        publish_latest_rich_state(updates, older)
        publish_latest_rich_state(updates, newer)

        latest = consume_latest_rich_state(updates, older)

        self.assertEqual(latest, newer)

    def test_quaternion_to_z_angle_recovers_planar_rotation(self) -> None:
        angle = math.pi / 4.0
        quaternion = (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))

        recovered = quaternion_to_z_angle(quaternion)

        self.assertAlmostEqual(recovered, angle)

    def test_decode_hammer_contact_state_reads_touching_and_normal(self) -> None:
        first_contact = struct.pack("<ffff", 27.7, 69.5, -0.9, -0.3)

        flags, normal = decode_hammer_contact_state(
            slide=True,
            contact_count=2,
            first_contact=first_contact,
        )

        self.assertEqual(flags, (1.0, 1.0))
        self.assertAlmostEqual(normal[0], -0.9)
        self.assertAlmostEqual(normal[1], -0.3)

    def test_format_payload_exposes_implemented_features(self) -> None:
        rich_state = RichStateSnapshot(
            ts=122.5,
            body_position_xy=(1.0, 2.0),
            body_velocity_xy=(3.0, 4.0),
            body_rotation_sin_cos=(0.5, 0.866),
            body_angular_velocity=1.25,
            hammer_anchor_xy=(5.0, 6.0),
            hammer_tip_xy=(7.0, 8.0),
            hammer_direction_sin_cos=(0.0, 1.0),
            hammer_angular_velocity=-2.5,
            hammer_contact_flags=(1.0, 0.0),
            hammer_contact_normal_xy=(-0.5, 0.75),
            progress_features=(13.0, 14.0, 15.0),
            valid=True,
        )
        payload = format_payload(
            ts=123.0,
            pid=456,
            addr=0xABC,
            rich_state=rich_state,
            cursor_position_xy=(9.0, 10.0),
            cursor_velocity_xy=(11.0, 12.0),
            previous_action=(16.0, 17.0),
        )

        self.assertEqual(
            payload["implemented_features"],
            [
                "body_position_xy",
                "body_velocity_xy",
                "body_rotation_sin_cos",
                "body_angular_velocity",
                "hammer_anchor_xy",
                "hammer_tip_xy",
                "hammer_direction_sin_cos",
                "hammer_angular_velocity",
                "cursor_position_xy",
                "cursor_velocity_xy",
                "hammer_contact_flags",
                "hammer_contact_normal_xy",
                "progress_features",
                "previous_action",
            ],
        )
        self.assertEqual(payload["body_position_xy"], [1.0, 2.0])
        self.assertEqual(payload["body_velocity_xy"], [3.0, 4.0])
        self.assertEqual(payload["hammer_tip_xy"], [7.0, 8.0])
        self.assertEqual(payload["hammer_contact_flags"], [1.0, 0.0])
        self.assertTrue(payload["rich_state_valid"])
        self.assertEqual(payload["rich_state_ts"], 122.5)
        self.assertEqual(payload["rich_state_age"], 0.5)
        self.assertEqual(payload["previous_action"], [16.0, 17.0])


if __name__ == "__main__":
    unittest.main()
