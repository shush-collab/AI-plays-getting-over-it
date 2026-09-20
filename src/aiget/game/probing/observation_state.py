#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import queue
import struct
import time
from dataclasses import dataclass, field
from threading import Event, Thread

from .live_layout import (
    ResolvedLiveLayout,
    default_live_layout_cache_path,
    load_live_layout,
    resolve_live_layout,
    save_live_layout,
    validate_live_layout,
)
from .live_position import freeze_fast_cursor_lane, log, read_fast_cursor_sample
from .memory_probe import MemReader, auto_pid, fmt_addr, read_optional_f32, read_optional_vec2

HAMMER_COLLISIONS_SLIDE_OFFSET = 0x4C
HAMMER_COLLISIONS_CONTACTS_OFFSET = 0x268
MANAGED_ARRAY_LENGTH_OFFSET = 0x18
MANAGED_ARRAY_DATA_OFFSET = 0x20
RICH_VALIDITY_FIELDS = (
    "body_position_xy",
    "hammer_anchor_xy",
    "hammer_tip_xy",
    "progress_features",
)


@dataclass(frozen=True)
class PositionSample:
    ts: float
    x: float
    y: float


@dataclass(frozen=True)
class AngleSample:
    ts: float
    angle: float


@dataclass(frozen=True)
class RichRawSample:
    ts: float
    body_position_xy: tuple[float, float] | None
    body_angle: float | None
    hammer_anchor_xy: tuple[float, float] | None
    hammer_tip_xy: tuple[float, float] | None
    hammer_contact_flags: tuple[float, float] | None
    hammer_contact_normal_xy: tuple[float, float] | None
    progress_features: tuple[float, float, float] | None
    valid_mask: dict[str, bool]
    layout_discovered_at: float


@dataclass(frozen=True)
class RawRichLane:
    pid: int
    refresh_interval: float
    layout: ResolvedLiveLayout


@dataclass(frozen=True)
class RichStateSnapshot:
    ts: float
    body_position_xy: tuple[float, float]
    body_velocity_xy: tuple[float, float]
    body_rotation_sin_cos: tuple[float, float]
    body_angular_velocity: float
    hammer_anchor_xy: tuple[float, float]
    hammer_tip_xy: tuple[float, float]
    hammer_direction_sin_cos: tuple[float, float]
    hammer_angular_velocity: float
    hammer_contact_flags: tuple[float, float]
    hammer_contact_normal_xy: tuple[float, float]
    progress_features: tuple[float, float, float]
    valid_mask: dict[str, bool] = field(default_factory=dict)
    source: str = "raw_memory"
    layout_discovered_at: float = 0.0
    valid: bool = True


@dataclass
class ProgressTracker:
    best_height: float
    last_progress_ts: float
    min_progress_delta: float = 1e-4

    def update(self, current_height: float, now: float) -> tuple[float, float, float]:
        if current_height > self.best_height + self.min_progress_delta:
            self.best_height = current_height
            self.last_progress_ts = now
        return current_height, self.best_height, max(0.0, now - self.last_progress_ts)


@dataclass
class SlowLaneAccumulator:
    tracker: ProgressTracker
    previous_body: PositionSample | None = None
    previous_body_angle: AngleSample | None = None
    previous_hammer_angle: AngleSample | None = None


def estimate_velocity(previous: PositionSample | None, current: PositionSample) -> tuple[float, float]:
    if previous is None:
        return 0.0, 0.0
    dt = current.ts - previous.ts
    if dt <= 0:
        return 0.0, 0.0
    return (current.x - previous.x) / dt, (current.y - previous.y) / dt


def wrap_angle_delta(delta: float) -> float:
    while delta <= -math.pi:
        delta += 2.0 * math.pi
    while delta > math.pi:
        delta -= 2.0 * math.pi
    return delta


def estimate_angular_velocity(previous: AngleSample | None, current: AngleSample) -> float:
    if previous is None:
        return 0.0
    dt = current.ts - previous.ts
    if dt <= 0:
        return 0.0
    return wrap_angle_delta(current.angle - previous.angle) / dt


def angle_to_sin_cos(angle: float) -> tuple[float, float]:
    return math.sin(angle), math.cos(angle)


def direction_to_angle(start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.atan2(dy, dx)


def decode_hammer_contact_state(
    *, slide: bool, contact_count: int, first_contact: bytes | None
) -> tuple[tuple[float, float], tuple[float, float]]:
    if contact_count <= 0 or first_contact is None:
        return (0.0, 1.0 if slide else 0.0), (0.0, 0.0)

    _, _, normal_x, normal_y = struct.unpack("<ffff", first_contact[:16])
    return (1.0, 1.0 if slide else 0.0), (normal_x, normal_y)


def freeze_raw_rich_lane(pid: int, refresh_interval: float, layout: ResolvedLiveLayout) -> RawRichLane:
    if layout.pid != pid:
        raise RuntimeError(f"ResolvedLiveLayout PID mismatch: layout has {layout.pid}, requested {pid}")
    return RawRichLane(pid=pid, refresh_interval=refresh_interval, layout=layout)


def empty_live_layout(pid: int, fast_cursor_addr: int, discovered_at: float | None = None) -> ResolvedLiveLayout:
    return ResolvedLiveLayout(
        pid=pid,
        fast_cursor_addr=fast_cursor_addr,
        body_position_addr=None,
        body_angle_addr=None,
        hammer_anchor_addr=None,
        hammer_tip_addr=None,
        hammer_contact_flags_addr=None,
        hammer_contact_normal_addr=None,
        progress_addr=None,
        valid_mask={
            "cursor_position_xy": True,
            "body_position_xy": False,
            "body_angle": False,
            "hammer_anchor_xy": False,
            "hammer_tip_xy": False,
            "hammer_contact_flags": False,
            "hammer_contact_normal_xy": False,
            "progress_features": False,
        },
        discovered_at=time.time() if discovered_at is None else discovered_at,
    )


def read_rich_raw_sample(reader: MemReader, lane: RawRichLane, ts: float | None = None) -> RichRawSample:
    sample_ts = time.time() if ts is None else ts
    valid_mask = _completed_valid_mask(lane.layout.valid_mask)

    progress_addr = lane.layout.progress_addr or lane.layout.body_position_addr
    fields = _read_grouped_memory(
        reader,
        {
            "body_position_xy": (lane.layout.body_position_addr, 8),
            "hammer_anchor_xy": (lane.layout.hammer_anchor_addr, 8),
            "hammer_tip_xy": (lane.layout.hammer_tip_addr, 8),
            "progress_xy": (progress_addr, 8),
        },
    )

    body_position_xy = _unpack_vec2_field(fields, valid_mask, "body_position_xy")
    hammer_anchor_xy = _unpack_vec2_field(fields, valid_mask, "hammer_anchor_xy")
    hammer_tip_xy = _unpack_vec2_field(fields, valid_mask, "hammer_tip_xy")
    progress_xy = _unpack_vec2_field(fields, valid_mask, "progress_features", source_name="progress_xy")
    progress_features = None if progress_xy is None else (progress_xy[1], progress_xy[1], 0.0)

    return RichRawSample(
        ts=sample_ts,
        body_position_xy=body_position_xy,
        body_angle=None,
        hammer_anchor_xy=hammer_anchor_xy,
        hammer_tip_xy=hammer_tip_xy,
        hammer_contact_flags=None,
        hammer_contact_normal_xy=None,
        progress_features=progress_features,
        valid_mask=valid_mask,
        layout_discovered_at=lane.layout.discovered_at,
    )


def build_rich_state_snapshot_from_raw(
    sample: RichRawSample,
    accumulator: SlowLaneAccumulator,
) -> RichStateSnapshot:
    valid_mask = _completed_valid_mask(sample.valid_mask)

    body_position_xy = sample.body_position_xy if valid_mask["body_position_xy"] and sample.body_position_xy is not None else (0.0, 0.0)
    body_velocity_xy = (0.0, 0.0)
    if valid_mask["body_position_xy"] and sample.body_position_xy is not None:
        body_current = PositionSample(ts=sample.ts, x=sample.body_position_xy[0], y=sample.body_position_xy[1])
        body_velocity_xy = estimate_velocity(accumulator.previous_body, body_current)
        accumulator.previous_body = body_current

    body_angle_valid = bool(valid_mask.get("body_angle", False))
    body_angle = sample.body_angle if body_angle_valid and sample.body_angle is not None else 0.0
    body_rotation_sin_cos = angle_to_sin_cos(body_angle)
    body_angular_velocity = 0.0
    if body_angle_valid and sample.body_angle is not None:
        body_angle_current = AngleSample(ts=sample.ts, angle=sample.body_angle)
        body_angular_velocity = estimate_angular_velocity(accumulator.previous_body_angle, body_angle_current)
        accumulator.previous_body_angle = body_angle_current

    hammer_anchor_xy = sample.hammer_anchor_xy if valid_mask["hammer_anchor_xy"] and sample.hammer_anchor_xy is not None else (0.0, 0.0)
    hammer_tip_xy = sample.hammer_tip_xy if valid_mask["hammer_tip_xy"] and sample.hammer_tip_xy is not None else (0.0, 0.0)
    hammer_direction_sin_cos = (0.0, 1.0)
    hammer_angular_velocity = 0.0
    if (
        valid_mask["hammer_anchor_xy"]
        and sample.hammer_anchor_xy is not None
        and valid_mask["hammer_tip_xy"]
        and sample.hammer_tip_xy is not None
    ):
        hammer_angle = direction_to_angle(sample.hammer_anchor_xy, sample.hammer_tip_xy)
        hammer_direction_sin_cos = angle_to_sin_cos(hammer_angle)
        hammer_angle_current = AngleSample(ts=sample.ts, angle=hammer_angle)
        hammer_angular_velocity = estimate_angular_velocity(accumulator.previous_hammer_angle, hammer_angle_current)
        accumulator.previous_hammer_angle = hammer_angle_current

    hammer_contact_flags = (
        sample.hammer_contact_flags
        if valid_mask.get("hammer_contact_flags", False) and sample.hammer_contact_flags is not None
        else (0.0, 0.0)
    )
    hammer_contact_normal_xy = (
        sample.hammer_contact_normal_xy
        if valid_mask.get("hammer_contact_normal_xy", False) and sample.hammer_contact_normal_xy is not None
        else (0.0, 0.0)
    )

    progress_features = (0.0, 0.0, 0.0)
    if valid_mask["progress_features"] and sample.progress_features is not None:
        progress_features = accumulator.tracker.update(current_height=sample.progress_features[0], now=sample.ts)

    return RichStateSnapshot(
        ts=sample.ts,
        body_position_xy=body_position_xy,
        body_velocity_xy=body_velocity_xy,
        body_rotation_sin_cos=body_rotation_sin_cos,
        body_angular_velocity=body_angular_velocity,
        hammer_anchor_xy=hammer_anchor_xy,
        hammer_tip_xy=hammer_tip_xy,
        hammer_direction_sin_cos=hammer_direction_sin_cos,
        hammer_angular_velocity=hammer_angular_velocity,
        hammer_contact_flags=hammer_contact_flags,
        hammer_contact_normal_xy=hammer_contact_normal_xy,
        progress_features=progress_features,
        valid_mask=valid_mask,
        layout_discovered_at=sample.layout_discovered_at,
        valid=any(valid_mask.values()),
    )


def publish_latest_rich_state(updates: queue.Queue[RichStateSnapshot], rich_state: RichStateSnapshot) -> None:
    while True:
        try:
            updates.put_nowait(rich_state)
            return
        except queue.Full:
            try:
                updates.get_nowait()
            except queue.Empty:
                continue


def consume_latest_rich_state(
    updates: queue.Queue[RichStateSnapshot], current: RichStateSnapshot
) -> RichStateSnapshot:
    latest = current
    while True:
        try:
            latest = updates.get_nowait()
        except queue.Empty:
            return latest


def run_slow_lane_worker(
    lane: RawRichLane,
    updates: queue.Queue[RichStateSnapshot],
    stop_event: Event,
    accumulator: SlowLaneAccumulator,
) -> None:
    refresh_interval = max(0.0, lane.refresh_interval)
    try:
        with MemReader(lane.pid) as reader:
            while not stop_event.is_set():
                rich_raw_sample = read_rich_raw_sample(reader, lane)
                rich_state = build_rich_state_snapshot_from_raw(rich_raw_sample, accumulator)
                publish_latest_rich_state(updates, rich_state)
                if stop_event.wait(refresh_interval):
                    break
    except Exception as exc:
        log(f"Slow observation lane refresh failed: {exc}")


def format_payload(
    *,
    ts: float,
    pid: int,
    addr: int,
    rich_state: RichStateSnapshot,
    cursor_position_xy: tuple[float, float],
    cursor_velocity_xy: tuple[float, float],
    previous_action: tuple[float, float],
) -> dict[str, object]:
    return {
        "ts": ts,
        "pid": pid,
        "addr": hex(addr),
        "implemented_features": [
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
        "body_position_xy": list(rich_state.body_position_xy),
        "body_velocity_xy": list(rich_state.body_velocity_xy),
        "body_rotation_sin_cos": list(rich_state.body_rotation_sin_cos),
        "body_angular_velocity": rich_state.body_angular_velocity,
        "hammer_anchor_xy": list(rich_state.hammer_anchor_xy),
        "hammer_tip_xy": list(rich_state.hammer_tip_xy),
        "hammer_direction_sin_cos": list(rich_state.hammer_direction_sin_cos),
        "hammer_angular_velocity": rich_state.hammer_angular_velocity,
        "cursor_position_xy": list(cursor_position_xy),
        "cursor_velocity_xy": list(cursor_velocity_xy),
        "hammer_contact_flags": list(rich_state.hammer_contact_flags),
        "hammer_contact_normal_xy": list(rich_state.hammer_contact_normal_xy),
        "progress_features": list(rich_state.progress_features),
        "previous_action": list(previous_action),
        "rich_state_valid": rich_state.valid,
        "rich_state_valid_mask": dict(rich_state.valid_mask),
        "rich_state_source": rich_state.source,
        "layout_discovered_at": rich_state.layout_discovered_at,
        "rich_state_ts": rich_state.ts,
        "rich_state_age": max(0.0, ts - rich_state.ts),
    }


def stream_observation_state(pid: int, args: argparse.Namespace) -> None:
    log(f"Resolving PlayerControl and calibrating fast cursor lane for PID {pid}.")
    fast_lane = freeze_fast_cursor_lane(
        pid=pid,
        calibration_samples=args.calibration_samples,
        calibration_interval=args.calibration_interval,
        window=args.window,
        eps=args.eps,
    )
    log(
        f"Frozen fast cursor lane {fast_lane.candidate.describe()} -> {fmt_addr(fast_lane.current_addr)} "
        f"(hits={fast_lane.calibration_hits}/{fast_lane.calibration_samples}, motion_span={fast_lane.motion_span:.6f})"
    )

    layout, layout_source = _resolve_or_load_live_layout(pid, args, fast_cursor_addr=fast_lane.current_addr)
    available_fields = sorted(key for key, value in layout.valid_mask.items() if value and key != "cursor_position_xy")
    missing_fields = sorted(key for key, value in layout.valid_mask.items() if not value and key != "cursor_position_xy")
    log(
        f"{layout_source} rich raw layout discovered at {layout.discovered_at:.6f}; "
        f"available={available_fields or ['none']} missing={missing_fields or ['none']}"
    )

    if args.validate_layout and any(layout.valid_mask.get(name, False) for name in RICH_VALIDITY_FIELDS):
        report = validate_live_layout(pid, layout)
        log(
            f"Live layout validation {'passed' if report.ok else 'failed'}; "
            f"mismatches={report.mismatches} notes={report.notes or ['none']}"
        )
    elif args.validate_layout:
        log("Skipping live layout validation because no rich raw fields were resolved.")

    raw_rich_lane = freeze_raw_rich_lane(pid, refresh_interval=args.rich_snapshot_interval, layout=layout)
    with MemReader(pid) as bootstrap_reader:
        initial_raw_sample = read_rich_raw_sample(bootstrap_reader, raw_rich_lane)

    initial_height = 0.0
    if initial_raw_sample.progress_features is not None:
        initial_height = initial_raw_sample.progress_features[0]
    accumulator = SlowLaneAccumulator(
        tracker=ProgressTracker(best_height=initial_height, last_progress_ts=initial_raw_sample.ts)
    )
    rich_state = build_rich_state_snapshot_from_raw(initial_raw_sample, accumulator)

    log(
        f"Streaming fast cursor lane at {args.interval:.3f}s intervals and refreshing raw rich-state lane every "
        f"{raw_rich_lane.refresh_interval:.3f}s."
    )

    rich_state_updates: queue.Queue[RichStateSnapshot] = queue.Queue(maxsize=1)
    stop_event = Event()
    slow_worker = Thread(
        target=run_slow_lane_worker,
        args=(raw_rich_lane, rich_state_updates, stop_event, accumulator),
        name="raw-rich-lane",
        daemon=True,
    )
    slow_worker.start()

    previous_cursor: PositionSample | None = None
    previous_action = tuple(args.previous_action)
    emitted = 0
    reader = MemReader(pid)
    try:
        while True:
            now = time.time()
            cursor_sample = read_fast_cursor_sample(reader, fast_lane, ts=now)
            rich_state = consume_latest_rich_state(rich_state_updates, rich_state)

            cursor_current = PositionSample(ts=now, x=cursor_sample.x, y=cursor_sample.y)
            cursor_velocity = estimate_velocity(previous_cursor, cursor_current)

            payload = format_payload(
                ts=now,
                pid=pid,
                addr=cursor_sample.addr,
                rich_state=rich_state,
                cursor_position_xy=(cursor_sample.x, cursor_sample.y),
                cursor_velocity_xy=cursor_velocity,
                previous_action=previous_action,
            )

            if args.format == "json":
                print(json.dumps(payload), flush=True)
            else:
                print(
                    f"{payload['ts']:.6f} pid={pid} addr={fmt_addr(cursor_sample.addr)} "
                    f"body=({rich_state.body_position_xy[0]:.6f}, {rich_state.body_position_xy[1]:.6f}) "
                    f"hammer_tip=({rich_state.hammer_tip_xy[0]:.6f}, {rich_state.hammer_tip_xy[1]:.6f}) "
                    f"cursor=({cursor_sample.x:.6f}, {cursor_sample.y:.6f}) "
                    f"progress=({rich_state.progress_features[0]:.6f}, {rich_state.progress_features[1]:.6f}, {rich_state.progress_features[2]:.6f}) "
                    f"valid={rich_state.valid_mask}",
                    flush=True,
                )

            previous_cursor = cursor_current
            emitted += 1
            if args.samples and emitted >= args.samples:
                break
            time.sleep(args.interval)
    finally:
        stop_event.set()
        slow_worker.join(timeout=1.0)
        reader.close()


def _completed_valid_mask(mask: dict[str, bool]) -> dict[str, bool]:
    completed = {name: False for name in RICH_VALIDITY_FIELDS}
    for key, value in mask.items():
        if key in completed:
            completed[key] = bool(value)
    return completed


def _read_grouped_memory(reader: MemReader, requests: dict[str, tuple[int | None, int]]) -> dict[str, bytes]:
    page_size = 4096
    groups: dict[int, dict[str, object]] = {}
    result: dict[str, bytes] = {}

    for name, (addr, size) in requests.items():
        if addr is None:
            continue
        group_key = addr // page_size
        group = groups.setdefault(group_key, {"start": addr, "end": addr + size, "fields": []})
        group["start"] = min(int(group["start"]), addr)
        group["end"] = max(int(group["end"]), addr + size)
        group["fields"].append((name, addr, size))

    for group in groups.values():
        start = int(group["start"])
        try:
            data = reader.read(start, int(group["end"]) - start)
        except OSError:
            continue
        for name, addr, size in group["fields"]:
            offset = addr - start
            result[name] = data[offset : offset + size]
    return result


def _unpack_vec2_field(
    fields: dict[str, bytes],
    valid_mask: dict[str, bool],
    field_name: str,
    *,
    source_name: str | None = None,
) -> tuple[float, float] | None:
    if not valid_mask[field_name]:
        return None
    raw = fields.get(source_name or field_name)
    if raw is None or len(raw) < 8:
        valid_mask[field_name] = False
        return None
    return struct.unpack("<ff", raw[:8])


def _read_optional_layout_vec2(
    reader: MemReader,
    addr: int | None,
    valid_mask: dict[str, bool],
    field_name: str,
) -> tuple[float, float] | None:
    if not valid_mask[field_name]:
        return None
    value = read_optional_vec2(reader, addr)
    if value is None:
        valid_mask[field_name] = False
    return value


def _read_optional_layout_f32(
    reader: MemReader,
    addr: int | None,
    valid_mask: dict[str, bool],
    field_name: str,
) -> float | None:
    if not valid_mask[field_name]:
        return None
    value = read_optional_f32(reader, addr)
    if value is None:
        valid_mask[field_name] = False
    return value


def _read_optional_hammer_contact_state(
    reader: MemReader,
    layout: ResolvedLiveLayout,
    valid_mask: dict[str, bool],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    flags_valid = valid_mask["hammer_contact_flags"]
    normal_valid = valid_mask["hammer_contact_normal_xy"]
    if not flags_valid and not normal_valid:
        return None, None
    if layout.hammer_contact_flags_addr is None or layout.hammer_contact_normal_addr is None:
        valid_mask["hammer_contact_flags"] = False
        valid_mask["hammer_contact_normal_xy"] = False
        return None, None

    try:
        slide = bool(reader.read_u8(layout.hammer_contact_flags_addr))
        contacts_obj = reader.read_ptr(layout.hammer_contact_normal_addr)
        if not contacts_obj:
            return decode_hammer_contact_state(slide=slide, contact_count=0, first_contact=None)
        contact_count = int(reader.read_ptr(contacts_obj + MANAGED_ARRAY_LENGTH_OFFSET))
        first_contact = reader.read(contacts_obj + MANAGED_ARRAY_DATA_OFFSET, 16) if contact_count > 0 else None
        return decode_hammer_contact_state(slide=slide, contact_count=contact_count, first_contact=first_contact)
    except OSError:
        valid_mask["hammer_contact_flags"] = False
        valid_mask["hammer_contact_normal_xy"] = False
        return None, None


def _read_optional_progress_features(
    reader: MemReader,
    layout: ResolvedLiveLayout,
    valid_mask: dict[str, bool],
) -> tuple[float, float, float] | None:
    if not valid_mask["progress_features"]:
        return None

    body_position = read_optional_vec2(reader, layout.progress_addr)
    if body_position is None:
        body_position = read_optional_vec2(reader, layout.body_position_addr)
    if body_position is None:
        valid_mask["progress_features"] = False
        return None
    return (body_position[1], body_position[1], 0.0)


def _resolve_or_load_live_layout(
    pid: int,
    args: argparse.Namespace,
    *,
    fast_cursor_addr: int,
) -> tuple[ResolvedLiveLayout, str]:
    cache_path = None if args.no_live_layout_cache else (args.live_layout_cache or default_live_layout_cache_path(pid))

    if cache_path:
        try:
            cached_layout = load_live_layout(cache_path)
        except FileNotFoundError:
            pass
        else:
            if cached_layout.pid == pid:
                return cached_layout, "Loaded"
            log(
                f"Ignoring cached live layout for PID {cached_layout.pid}; current process is PID {pid}. "
                "Rediscovering."
            )

    try:
        layout = resolve_live_layout(
            pid,
            calibration_samples=args.calibration_samples,
            calibration_interval=args.calibration_interval,
            window=args.window,
            eps=args.eps,
            startup_timeout=args.layout_discovery_timeout,
            resolve_optional_fields=args.resolve_optional_rich_fields,
            fast_cursor_addr=fast_cursor_addr,
        )
    except Exception as exc:
        log(f"Rich raw layout discovery failed fast; falling back to fast-lane-only startup: {exc}")
        layout = empty_live_layout(pid, fast_cursor_addr)
        source = "Fallback"
    else:
        source = "Resolved"

    if cache_path:
        save_live_layout(cache_path, layout)
    return layout, source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream the implemented observation-state features for Getting Over It using raw memory after startup discovery."
    )
    parser.add_argument("--pid", type=int, default=None, help="Target PID. Defaults to pgrep.")
    parser.add_argument("--interval", type=float, default=0.1, help="Seconds between emitted samples.")
    parser.add_argument(
        "--rich-snapshot-interval",
        dest="rich_snapshot_interval",
        type=float,
        default=0.5,
        help="Seconds between raw rich-state snapshots in the background worker.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=1,
        help="Number of startup discovery samples used to choose raw-memory lanes.",
    )
    parser.add_argument(
        "--calibration-interval",
        type=float,
        default=0.7,
        help="Seconds between startup discovery samples.",
    )
    parser.add_argument(
        "--window",
        type=lambda value: int(value, 0),
        default=0x200,
        help="Bytes around each discovery root/pointee to scan for matching values.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0015,
        help="Float tolerance when matching raw memory to authoritative startup samples.",
    )
    parser.add_argument(
        "--live-layout-cache",
        type=str,
        default=None,
        help="Optional JSON file for saving/loading the resolved raw rich-state layout. Defaults to a per-PID cache under ~/.cache/aiget/.",
    )
    parser.add_argument(
        "--no-live-layout-cache",
        action="store_true",
        help="Disable cache-first rich layout startup and do not save the resolved layout.",
    )
    parser.add_argument(
        "--layout-discovery-timeout",
        type=float,
        default=1.5,
        help="Maximum seconds to spend resolving the raw rich-state layout before returning a partial result.",
    )
    parser.add_argument(
        "--resolve-optional-rich-fields",
        action="store_true",
        help="Also try to resolve optional rich raw fields such as body angle and hammer anchor/tip at startup.",
    )
    parser.add_argument(
        "--validate-layout",
        action="store_true",
        help="Validate the resolved live layout once at startup using authoritative ptrace samples.",
    )
    parser.add_argument(
        "--previous-action",
        nargs=2,
        type=float,
        metavar=("AX", "AY"),
        default=(0.0, 0.0),
        help="Previous action values to echo in the observation payload.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of payloads to emit before exiting. Use 0 for an indefinite stream.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="json", help="Output format.")
    args = parser.parse_args()

    pid = args.pid if args.pid is not None else auto_pid()
    stream_observation_state(pid, args)


if __name__ == "__main__":
    main()
