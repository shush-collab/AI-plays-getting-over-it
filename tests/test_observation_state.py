import math
import queue
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.live_layout import ResolvedLiveLayout
from aiget.observation_state import (
    AngleSample,
    PositionSample,
    ProgressTracker,
    RawRichLane,
    RichRawSample,
    RichStateSnapshot,
    SlowLaneAccumulator,
    build_rich_state_snapshot_from_raw,
    consume_latest_rich_state,
    decode_hammer_contact_state,
    estimate_angular_velocity,
    estimate_velocity,
    format_payload,
    freeze_raw_rich_lane,
    empty_live_layout,
    publish_latest_rich_state,
    read_rich_raw_sample,
    run_slow_lane_worker,
    _resolve_or_load_live_layout,
)


class _FakeReader:
    def __init__(
        self,
        *,
        vec2: dict[int, tuple[float, float]] | None = None,
        f32: dict[int, float] | None = None,
        u8: dict[int, int] | None = None,
        ptr: dict[int, int] | None = None,
        raw: dict[int, bytes] | None = None,
    ):
        self.vec2 = vec2 or {}
        self.f32 = f32 or {}
        self.u8 = u8 or {}
        self.ptr = ptr or {}
        self.raw = raw or {}

    def read_vec2(self, addr: int) -> tuple[float, float]:
        return self.vec2[addr]

    def read_f32(self, addr: int) -> float:
        return self.f32[addr]

    def read_u8(self, addr: int) -> int:
        return self.u8[addr]

    def read_ptr(self, addr: int) -> int:
        return self.ptr[addr]

    def read(self, addr: int, size: int) -> bytes:
        data = self.raw[addr]
        return data[:size]


class _FakeReaderContext(_FakeReader):
    def __enter__(self) -> "_FakeReaderContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _OneShotStopEvent:
    def __init__(self) -> None:
        self._set = False
        self.wait_calls: list[float] = []

    def is_set(self) -> bool:
        return self._set

    def wait(self, timeout: float) -> bool:
        self.wait_calls.append(timeout)
        self._set = True
        return True


class ObservationStateTests(unittest.TestCase):
    def test_empty_live_layout_marks_all_rich_fields_invalid(self) -> None:
        layout = empty_live_layout(pid=1234, fast_cursor_addr=0xABC, discovered_at=10.0)

        self.assertEqual(layout.fast_cursor_addr, 0xABC)
        self.assertTrue(layout.valid_mask["cursor_position_xy"])
        self.assertFalse(layout.valid_mask["body_position_xy"])
        self.assertFalse(layout.valid_mask["hammer_tip_xy"])

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

    def test_freeze_raw_rich_lane_keeps_layout_and_refresh_interval(self) -> None:
        layout = ResolvedLiveLayout(
            pid=1234,
            fast_cursor_addr=0x1000,
            body_position_addr=0x2000,
            body_angle_addr=None,
            hammer_anchor_addr=0x3000,
            hammer_tip_addr=0x4000,
            hammer_contact_flags_addr=0x5000,
            hammer_contact_normal_addr=0x6000,
            progress_addr=0x2000,
            valid_mask={"body_position_xy": True},
            discovered_at=55.0,
        )

        lane = freeze_raw_rich_lane(pid=1234, refresh_interval=0.75, layout=layout)

        self.assertEqual(lane.pid, 1234)
        self.assertEqual(lane.layout, layout)
        self.assertEqual(lane.refresh_interval, 0.75)

    def test_read_rich_raw_sample_reads_only_resolved_raw_addresses(self) -> None:
        layout = ResolvedLiveLayout(
            pid=1234,
            fast_cursor_addr=0x1000,
            body_position_addr=0x2000,
            body_angle_addr=0x3000,
            hammer_anchor_addr=0x4000,
            hammer_tip_addr=0x5000,
            hammer_contact_flags_addr=0x6000,
            hammer_contact_normal_addr=0x7000,
            progress_addr=0x2000,
            valid_mask={
                "body_position_xy": True,
                "body_angle": True,
                "hammer_anchor_xy": True,
                "hammer_tip_xy": True,
                "hammer_contact_flags": True,
                "hammer_contact_normal_xy": True,
                "progress_features": True,
            },
            discovered_at=77.0,
        )
        lane = RawRichLane(pid=1234, refresh_interval=0.2, layout=layout)
        first_contact = struct.pack("<ffff", 1.0, 2.0, -0.5, 0.75)
        reader = _FakeReader(
            vec2={
                0x2000: (6.0, 7.0),
                0x4000: (2.0, 3.0),
                0x5000: (4.0, 6.0),
            },
            f32={0x3000: math.pi / 4.0},
            u8={0x6000: 1},
            ptr={0x7000: 0x8000, 0x8000 + 0x18: 2},
            raw={0x8000 + 0x20: first_contact},
        )

        sample = read_rich_raw_sample(reader, lane, ts=101.0)

        self.assertEqual(sample.ts, 101.0)
        self.assertEqual(sample.body_position_xy, (6.0, 7.0))
        self.assertAlmostEqual(sample.body_angle or 0.0, math.pi / 4.0)
        self.assertEqual(sample.hammer_anchor_xy, (2.0, 3.0))
        self.assertEqual(sample.hammer_tip_xy, (4.0, 6.0))
        self.assertEqual(sample.hammer_contact_flags, (1.0, 1.0))
        self.assertEqual(sample.hammer_contact_normal_xy, (-0.5, 0.75))
        self.assertEqual(sample.progress_features, (7.0, 7.0, 0.0))
        self.assertTrue(all(sample.valid_mask.values()))
        self.assertEqual(sample.layout_discovered_at, 77.0)

    def test_build_rich_state_snapshot_from_raw_batches_fields_and_tracks_progress(self) -> None:
        accumulator = SlowLaneAccumulator(
            tracker=ProgressTracker(best_height=5.0, last_progress_ts=100.0),
            previous_body=PositionSample(ts=100.0, x=4.0, y=5.0),
            previous_body_angle=AngleSample(ts=100.0, angle=0.0),
            previous_hammer_angle=AngleSample(ts=100.0, angle=0.0),
        )
        sample = RichRawSample(
            ts=101.0,
            body_position_xy=(6.0, 7.0),
            body_angle=math.pi / 4.0,
            hammer_anchor_xy=(2.0, 3.0),
            hammer_tip_xy=(4.0, 6.0),
            hammer_contact_flags=(1.0, 0.0),
            hammer_contact_normal_xy=(-0.5, 0.75),
            progress_features=(7.0, 7.0, 0.0),
            valid_mask={
                "body_position_xy": True,
                "body_angle": True,
                "hammer_anchor_xy": True,
                "hammer_tip_xy": True,
                "hammer_contact_flags": True,
                "hammer_contact_normal_xy": True,
                "progress_features": True,
            },
            layout_discovered_at=88.0,
        )

        rich_snapshot = build_rich_state_snapshot_from_raw(sample, accumulator)

        self.assertEqual(rich_snapshot.ts, 101.0)
        self.assertEqual(rich_snapshot.body_position_xy, (6.0, 7.0))
        self.assertEqual(rich_snapshot.body_velocity_xy, (2.0, 2.0))
        self.assertAlmostEqual(rich_snapshot.body_rotation_sin_cos[0], math.sin(math.pi / 4.0))
        self.assertAlmostEqual(rich_snapshot.body_rotation_sin_cos[1], math.cos(math.pi / 4.0))
        self.assertEqual(rich_snapshot.hammer_anchor_xy, (2.0, 3.0))
        self.assertEqual(rich_snapshot.hammer_tip_xy, (4.0, 6.0))
        self.assertEqual(rich_snapshot.hammer_contact_flags, (1.0, 0.0))
        self.assertEqual(rich_snapshot.progress_features, (7.0, 7.0, 0.0))
        self.assertEqual(rich_snapshot.source, "raw_memory")
        self.assertEqual(rich_snapshot.layout_discovered_at, 88.0)
        self.assertTrue(rich_snapshot.valid)
        self.assertEqual(accumulator.previous_body, PositionSample(ts=101.0, x=6.0, y=7.0))
        self.assertEqual(accumulator.previous_body_angle, AngleSample(ts=101.0, angle=math.pi / 4.0))
        self.assertAlmostEqual(accumulator.previous_hammer_angle.angle, math.atan2(3.0, 2.0))

    def test_invalid_rich_fields_still_serialize_cleanly(self) -> None:
        accumulator = SlowLaneAccumulator(tracker=ProgressTracker(best_height=0.0, last_progress_ts=0.0))
        sample = RichRawSample(
            ts=25.0,
            body_position_xy=None,
            body_angle=None,
            hammer_anchor_xy=None,
            hammer_tip_xy=None,
            hammer_contact_flags=None,
            hammer_contact_normal_xy=None,
            progress_features=None,
            valid_mask={
                "body_position_xy": False,
                "body_angle": False,
                "hammer_anchor_xy": False,
                "hammer_tip_xy": False,
                "hammer_contact_flags": False,
                "hammer_contact_normal_xy": False,
                "progress_features": False,
            },
            layout_discovered_at=12.0,
        )
        rich_state = build_rich_state_snapshot_from_raw(sample, accumulator)

        payload = format_payload(
            ts=26.0,
            pid=456,
            addr=0xABC,
            rich_state=rich_state,
            cursor_position_xy=(9.0, 10.0),
            cursor_velocity_xy=(11.0, 12.0),
            previous_action=(16.0, 17.0),
        )

        self.assertEqual(payload["body_position_xy"], [0.0, 0.0])
        self.assertEqual(payload["hammer_tip_xy"], [0.0, 0.0])
        self.assertEqual(payload["progress_features"], [0.0, 0.0, 0.0])
        self.assertFalse(payload["rich_state_valid"])
        self.assertEqual(
            payload["rich_state_valid_mask"],
            {
                "body_position_xy": False,
                "body_angle": False,
                "hammer_anchor_xy": False,
                "hammer_tip_xy": False,
                "hammer_contact_flags": False,
                "hammer_contact_normal_xy": False,
                "progress_features": False,
            },
        )
        self.assertEqual(payload["rich_state_source"], "raw_memory")
        self.assertEqual(payload["layout_discovered_at"], 12.0)

    def test_publish_and_consume_latest_rich_state_reuses_newest_snapshot(self) -> None:
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
            valid_mask={name: True for name in ("body_position_xy", "body_angle", "hammer_anchor_xy", "hammer_tip_xy", "hammer_contact_flags", "hammer_contact_normal_xy", "progress_features")},
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
            valid_mask={name: True for name in ("body_position_xy", "body_angle", "hammer_anchor_xy", "hammer_tip_xy", "hammer_contact_flags", "hammer_contact_normal_xy", "progress_features")},
        )

        publish_latest_rich_state(updates, older)
        publish_latest_rich_state(updates, newer)

        latest = consume_latest_rich_state(updates, older)

        self.assertEqual(latest, newer)

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

    def test_format_payload_exposes_rich_state_age_and_validity_metadata(self) -> None:
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
            valid_mask={name: True for name in ("body_position_xy", "body_angle", "hammer_anchor_xy", "hammer_tip_xy", "hammer_contact_flags", "hammer_contact_normal_xy", "progress_features")},
            layout_discovered_at=98.0,
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

        self.assertTrue(payload["rich_state_valid"])
        self.assertEqual(payload["rich_state_ts"], 122.5)
        self.assertEqual(payload["rich_state_age"], 0.5)
        self.assertEqual(payload["rich_state_source"], "raw_memory")
        self.assertEqual(payload["layout_discovered_at"], 98.0)
        self.assertEqual(payload["previous_action"], [16.0, 17.0])

    def test_run_slow_lane_worker_uses_raw_reader_path_only(self) -> None:
        lane = RawRichLane(
            pid=1234,
            refresh_interval=0.2,
            layout=ResolvedLiveLayout(
                pid=1234,
                fast_cursor_addr=0x1000,
                body_position_addr=0x2000,
                body_angle_addr=None,
                hammer_anchor_addr=0x3000,
                hammer_tip_addr=0x4000,
                hammer_contact_flags_addr=0x5000,
                hammer_contact_normal_addr=0x6000,
                progress_addr=0x2000,
                valid_mask={"body_position_xy": True},
                discovered_at=33.0,
            ),
        )
        fake_reader = _FakeReaderContext()
        stop_event = _OneShotStopEvent()
        updates: queue.Queue[RichStateSnapshot] = queue.Queue(maxsize=1)
        accumulator = SlowLaneAccumulator(tracker=ProgressTracker(best_height=0.0, last_progress_ts=0.0))
        rich_sample = RichRawSample(
            ts=50.0,
            body_position_xy=(1.0, 2.0),
            body_angle=None,
            hammer_anchor_xy=None,
            hammer_tip_xy=None,
            hammer_contact_flags=None,
            hammer_contact_normal_xy=None,
            progress_features=(2.0, 2.0, 0.0),
            valid_mask={
                "body_position_xy": True,
                "body_angle": False,
                "hammer_anchor_xy": False,
                "hammer_tip_xy": False,
                "hammer_contact_flags": False,
                "hammer_contact_normal_xy": False,
                "progress_features": True,
            },
            layout_discovered_at=33.0,
        )
        rich_state = build_rich_state_snapshot_from_raw(rich_sample, accumulator)

        with (
            patch("aiget.observation_state.MemReader", return_value=fake_reader) as mem_reader_cls,
            patch("aiget.observation_state.read_rich_raw_sample", return_value=rich_sample) as read_raw,
            patch("aiget.observation_state.build_rich_state_snapshot_from_raw", return_value=rich_state) as build_snapshot,
        ):
            run_slow_lane_worker(lane, updates, stop_event, accumulator)

        mem_reader_cls.assert_called_once_with(1234)
        read_raw.assert_called_once_with(fake_reader, lane)
        build_snapshot.assert_called_once_with(rich_sample, accumulator)
        self.assertEqual(updates.get_nowait(), rich_state)

    def test_resolve_or_load_live_layout_prefers_cache_for_same_pid(self) -> None:
        args = type(
            "Args",
            (),
            {
                "no_live_layout_cache": False,
                "live_layout_cache": "/tmp/layout.json",
                "calibration_samples": 1,
                "calibration_interval": 0.7,
                "window": 0x200,
                "eps": 0.0015,
                "layout_discovery_timeout": 1.5,
                "resolve_optional_rich_fields": False,
            },
        )()
        cached_layout = ResolvedLiveLayout(
            pid=1234,
            fast_cursor_addr=0xAAA,
            body_position_addr=0xBBB,
            body_angle_addr=None,
            hammer_anchor_addr=None,
            hammer_tip_addr=None,
            hammer_contact_flags_addr=None,
            hammer_contact_normal_addr=None,
            progress_addr=0xBBB,
            valid_mask={"cursor_position_xy": True, "body_position_xy": True},
            discovered_at=15.0,
        )

        with (
            patch("aiget.observation_state.load_live_layout", return_value=cached_layout),
            patch("aiget.observation_state.resolve_live_layout") as resolve_layout,
        ):
            layout, source = _resolve_or_load_live_layout(1234, args, fast_cursor_addr=0xABC)

        self.assertEqual(layout, cached_layout)
        self.assertEqual(source, "Loaded")
        resolve_layout.assert_not_called()

    def test_resolve_or_load_live_layout_falls_back_to_empty_layout_on_failure(self) -> None:
        args = type(
            "Args",
            (),
            {
                "no_live_layout_cache": True,
                "live_layout_cache": None,
                "calibration_samples": 1,
                "calibration_interval": 0.7,
                "window": 0x200,
                "eps": 0.0015,
                "layout_discovery_timeout": 1.5,
                "resolve_optional_rich_fields": False,
            },
        )()

        with patch("aiget.observation_state.resolve_live_layout", side_effect=RuntimeError("timeout")):
            layout, source = _resolve_or_load_live_layout(1234, args, fast_cursor_addr=0xABC)

        self.assertEqual(source, "Fallback")
        self.assertEqual(layout.fast_cursor_addr, 0xABC)
        self.assertFalse(layout.valid_mask["body_position_xy"])


if __name__ == "__main__":
    unittest.main()
