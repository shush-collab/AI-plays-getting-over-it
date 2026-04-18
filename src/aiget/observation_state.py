#!/usr/bin/env python3
import argparse
import json
import math
import queue
import struct
import time
from dataclasses import dataclass
from threading import Event, Thread

from .live_position import freeze_fast_cursor_lane, log, read_fast_cursor_sample
from .memory_probe import MemReader, auto_pid, fmt_addr
from .ptrace_il2cpp import ICALL_TRANSFORM_GET_POSITION, Il2CppRemote, RemoteError, RemoteProcess

ICALL_TRANSFORM_GET_ROTATION = "UnityEngine.Transform::get_rotation_Injected(UnityEngine.Quaternion&)"

HAMMER_COLLISIONS_SLIDE_OFFSET = 0x4C
HAMMER_COLLISIONS_CONTACTS_OFFSET = 0x268
MANAGED_ARRAY_LENGTH_OFFSET = 0x18
MANAGED_ARRAY_DATA_OFFSET = 0x20


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
class ObservationRefs:
    player_obj: int
    fake_cursor_obj: int
    fake_cursor_native: int
    fake_cursor_rb_obj: int
    fake_cursor_rb_native: int
    body_transform_obj: int
    hammer_anchor_transform_obj: int
    hammer_tip_transform_obj: int
    hammer_collisions_obj: int


@dataclass(frozen=True)
class ObservationSnapshot:
    body_position_xy: tuple[float, float]
    body_angle: float
    hammer_anchor_xy: tuple[float, float]
    hammer_tip_xy: tuple[float, float]
    hammer_contact_flags: tuple[float, float]
    hammer_contact_normal_xy: tuple[float, float]


@dataclass(frozen=True)
class SlowObservationLane:
    pid: int
    refs: ObservationRefs
    refresh_interval: float


@dataclass(frozen=True)
class SlowObservationSample:
    ts: float
    snapshot: ObservationSnapshot


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


def quaternion_to_z_angle(quaternion: tuple[float, float, float, float]) -> float:
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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


def _require_object_field(runtime: Il2CppRemote, klass: int, obj: int, field_name: str, owner_name: str) -> int:
    field = runtime.field_from_name(klass, field_name)
    if not field:
        raise RemoteError(f"{owner_name}.{field_name} field lookup failed")
    value = runtime.get_field_value_ptr(obj, field)
    if not value:
        raise RemoteError(f"{owner_name}.{field_name} resolved to null")
    return value


def resolve_observation_refs(pid: int) -> ObservationRefs:
    process = RemoteProcess(pid)
    try:
        process.attach_all()
        runtime = Il2CppRemote(process)

        domain = runtime.call("il2cpp_domain_get", [])
        runtime.call("il2cpp_thread_attach", [domain])
        images = runtime.find_images()

        assembly_csharp = images.get("Assembly-CSharp.dll")
        unity_core = images.get("UnityEngine.CoreModule.dll")
        if not assembly_csharp or not unity_core:
            raise RemoteError(f"Required images not found. Available: {', '.join(sorted(images))}")

        player_class = runtime.class_from_name(assembly_csharp, "", "PlayerControl")
        pose_class = runtime.class_from_name(assembly_csharp, "", "PoseControl")
        object_class = runtime.class_from_name(unity_core, "UnityEngine", "Object")
        if not player_class or not pose_class or not object_class:
            raise RemoteError("PlayerControl, PoseControl, or UnityEngine.Object class not found")

        player_type = runtime.call("il2cpp_class_get_type", [player_class])
        player_type_obj = runtime.call("il2cpp_type_get_object", [player_type])
        find_object = runtime.method_from_name(object_class, "FindObjectOfType", 1)
        if not find_object:
            raise RemoteError("UnityEngine.Object.FindObjectOfType(Type) method not found")

        player_obj = runtime.invoke(find_object, 0, [player_type_obj])
        if not player_obj:
            raise RemoteError("FindObjectOfType(PlayerControl) returned null")

        fake_cursor_obj = _require_object_field(runtime, player_class, player_obj, "fakeCursor", "PlayerControl")
        fake_cursor_rb_obj = _require_object_field(runtime, player_class, player_obj, "fakeCursorRB", "PlayerControl")
        pose_obj = _require_object_field(runtime, player_class, player_obj, "pose", "PlayerControl")
        hammer_collisions_obj = _require_object_field(
            runtime, player_class, player_obj, "hammerCollisions", "PlayerControl"
        )

        body_transform_obj = _require_object_field(runtime, pose_class, pose_obj, "potMeshHub", "PoseControl")
        hammer_anchor_transform_obj = _require_object_field(runtime, pose_class, pose_obj, "handle", "PoseControl")
        hammer_tip_transform_obj = _require_object_field(runtime, pose_class, pose_obj, "tip", "PoseControl")

        fake_cursor_native = process.read_ptr(fake_cursor_obj + 0x10)
        fake_cursor_rb_native = process.read_ptr(fake_cursor_rb_obj + 0x10)

        return ObservationRefs(
            player_obj=player_obj,
            fake_cursor_obj=fake_cursor_obj,
            fake_cursor_native=fake_cursor_native,
            fake_cursor_rb_obj=fake_cursor_rb_obj,
            fake_cursor_rb_native=fake_cursor_rb_native,
            body_transform_obj=body_transform_obj,
            hammer_anchor_transform_obj=hammer_anchor_transform_obj,
            hammer_tip_transform_obj=hammer_tip_transform_obj,
            hammer_collisions_obj=hammer_collisions_obj,
        )
    finally:
        process.close()


def _sample_transform_position(runtime: Il2CppRemote, transform_obj: int) -> tuple[float, float, float]:
    fn = runtime.resolve_icall(ICALL_TRANSFORM_GET_POSITION)
    runtime.process.reset_call_frame()
    out_ptr = runtime.process.alloc(b"\x00" * 12)
    runtime.process.execute(fn, [transform_obj, out_ptr])
    return struct.unpack("<fff", runtime.process.read(out_ptr, 12))


def _sample_transform_rotation(runtime: Il2CppRemote, transform_obj: int) -> tuple[float, float, float, float]:
    fn = runtime.resolve_icall(ICALL_TRANSFORM_GET_ROTATION)
    runtime.process.reset_call_frame()
    out_ptr = runtime.process.alloc(b"\x00" * 16)
    runtime.process.execute(fn, [transform_obj, out_ptr])
    return struct.unpack("<ffff", runtime.process.read(out_ptr, 16))


def _sample_hammer_contact_state(process: RemoteProcess, hammer_collisions_obj: int) -> tuple[tuple[float, float], tuple[float, float]]:
    slide = bool(process.read(hammer_collisions_obj + HAMMER_COLLISIONS_SLIDE_OFFSET, 1)[0])
    contacts_obj = process.read_ptr(hammer_collisions_obj + HAMMER_COLLISIONS_CONTACTS_OFFSET)
    if not contacts_obj:
        return decode_hammer_contact_state(slide=slide, contact_count=0, first_contact=None)

    contact_count = int(process.read_ptr(contacts_obj + MANAGED_ARRAY_LENGTH_OFFSET))
    first_contact = None
    if contact_count > 0:
        first_contact = process.read(contacts_obj + MANAGED_ARRAY_DATA_OFFSET, 16)
    return decode_hammer_contact_state(slide=slide, contact_count=contact_count, first_contact=first_contact)


def sample_observation_snapshot(pid: int, refs: ObservationRefs) -> ObservationSnapshot:
    process = RemoteProcess(pid)
    try:
        process.attach_all()
        runtime = Il2CppRemote(process)

        body_position = _sample_transform_position(runtime, refs.body_transform_obj)
        hammer_anchor = _sample_transform_position(runtime, refs.hammer_anchor_transform_obj)
        hammer_tip = _sample_transform_position(runtime, refs.hammer_tip_transform_obj)
        body_rotation = _sample_transform_rotation(runtime, refs.body_transform_obj)
        hammer_contact_flags, hammer_contact_normal = _sample_hammer_contact_state(process, refs.hammer_collisions_obj)

        return ObservationSnapshot(
            body_position_xy=(body_position[0], body_position[1]),
            body_angle=quaternion_to_z_angle(body_rotation),
            hammer_anchor_xy=(hammer_anchor[0], hammer_anchor[1]),
            hammer_tip_xy=(hammer_tip[0], hammer_tip[1]),
            hammer_contact_flags=hammer_contact_flags,
            hammer_contact_normal_xy=hammer_contact_normal,
        )
    finally:
        process.close()


def freeze_slow_observation_lane(pid: int, refresh_interval: float) -> SlowObservationLane:
    return SlowObservationLane(
        pid=pid,
        refs=resolve_observation_refs(pid),
        refresh_interval=refresh_interval,
    )


def read_slow_observation_sample(lane: SlowObservationLane, ts: float | None = None) -> SlowObservationSample:
    sample_ts = time.time() if ts is None else ts
    snapshot = sample_observation_snapshot(lane.pid, lane.refs)
    return SlowObservationSample(ts=sample_ts, snapshot=snapshot)


def build_rich_state_snapshot(
    sample: SlowObservationSample,
    accumulator: SlowLaneAccumulator,
) -> RichStateSnapshot:
    snapshot = sample.snapshot
    body_current = PositionSample(
        ts=sample.ts,
        x=snapshot.body_position_xy[0],
        y=snapshot.body_position_xy[1],
    )
    body_angle_current = AngleSample(ts=sample.ts, angle=snapshot.body_angle)
    hammer_angle = direction_to_angle(snapshot.hammer_anchor_xy, snapshot.hammer_tip_xy)
    hammer_angle_current = AngleSample(ts=sample.ts, angle=hammer_angle)

    rich_snapshot = RichStateSnapshot(
        ts=sample.ts,
        body_position_xy=snapshot.body_position_xy,
        body_velocity_xy=estimate_velocity(accumulator.previous_body, body_current),
        body_rotation_sin_cos=angle_to_sin_cos(snapshot.body_angle),
        body_angular_velocity=estimate_angular_velocity(accumulator.previous_body_angle, body_angle_current),
        hammer_anchor_xy=snapshot.hammer_anchor_xy,
        hammer_tip_xy=snapshot.hammer_tip_xy,
        hammer_direction_sin_cos=angle_to_sin_cos(hammer_angle),
        hammer_angular_velocity=estimate_angular_velocity(accumulator.previous_hammer_angle, hammer_angle_current),
        hammer_contact_flags=snapshot.hammer_contact_flags,
        hammer_contact_normal_xy=snapshot.hammer_contact_normal_xy,
        progress_features=accumulator.tracker.update(current_height=snapshot.body_position_xy[1], now=sample.ts),
    )
    accumulator.previous_body = body_current
    accumulator.previous_body_angle = body_angle_current
    accumulator.previous_hammer_angle = hammer_angle_current
    return rich_snapshot


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
    lane: SlowObservationLane,
    updates: queue.Queue[RichStateSnapshot],
    stop_event: Event,
    accumulator: SlowLaneAccumulator,
) -> None:
    refresh_interval = max(0.0, lane.refresh_interval)
    while not stop_event.wait(refresh_interval):
        try:
            slow_sample = read_slow_observation_sample(lane)
            rich_state = build_rich_state_snapshot(slow_sample, accumulator)
            publish_latest_rich_state(updates, rich_state)
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
        "rich_state_ts": rich_state.ts,
        "rich_state_age": max(0.0, ts - rich_state.ts),
    }


def stream_observation_state(pid: int, args: argparse.Namespace) -> None:
    log(f"Resolving PlayerControl and calibrating observation state path for PID {pid}.")
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

    slow_lane = freeze_slow_observation_lane(pid, refresh_interval=args.unity_snapshot_interval)
    initial_slow_sample = read_slow_observation_sample(slow_lane)
    initial_snapshot = initial_slow_sample.snapshot
    accumulator = SlowLaneAccumulator(
        tracker=ProgressTracker(best_height=initial_snapshot.body_position_xy[1], last_progress_ts=time.time())
    )
    log(
        f"Streaming fast cursor lane at {args.interval:.3f}s intervals and refreshing slow Unity lane in the background every "
        f"{slow_lane.refresh_interval:.3f}s."
    )

    previous_cursor: PositionSample | None = None

    rich_state = build_rich_state_snapshot(initial_slow_sample, accumulator)
    rich_state_updates: queue.Queue[RichStateSnapshot] = queue.Queue(maxsize=1)
    stop_event = Event()
    slow_worker = Thread(
        target=run_slow_lane_worker,
        args=(slow_lane, rich_state_updates, stop_event, accumulator),
        name="slow-observation-lane",
        daemon=True,
    )
    slow_worker.start()

    reader = MemReader(pid)
    previous_action = tuple(args.previous_action)
    emitted = 0
    try:
        while True:
            now = time.time()
            cursor_sample = read_fast_cursor_sample(reader, fast_lane, ts=now)
            addr = cursor_sample.addr
            cursor_x = cursor_sample.x
            cursor_y = cursor_sample.y

            rich_state = consume_latest_rich_state(rich_state_updates, rich_state)

            cursor_current = PositionSample(ts=now, x=cursor_x, y=cursor_y)

            cursor_velocity = estimate_velocity(previous_cursor, cursor_current)

            payload = format_payload(
                ts=now,
                pid=pid,
                addr=addr,
                rich_state=rich_state,
                cursor_position_xy=(cursor_x, cursor_y),
                cursor_velocity_xy=cursor_velocity,
                previous_action=previous_action,
            )

            if args.format == "json":
                print(json.dumps(payload), flush=True)
            else:
                print(
                    f"{payload['ts']:.6f} pid={pid} addr={fmt_addr(addr)} "
                    f"body=({rich_state.body_position_xy[0]:.6f}, {rich_state.body_position_xy[1]:.6f}) "
                    f"hammer_tip=({rich_state.hammer_tip_xy[0]:.6f}, {rich_state.hammer_tip_xy[1]:.6f}) "
                    f"cursor=({cursor_x:.6f}, {cursor_y:.6f}) "
                    f"progress=({rich_state.progress_features[0]:.6f}, {rich_state.progress_features[1]:.6f}, {rich_state.progress_features[2]:.6f})",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream the currently implemented observation-state features for Getting Over It."
    )
    parser.add_argument("--pid", type=int, default=None, help="Target PID. Defaults to pgrep.")
    parser.add_argument("--interval", type=float, default=0.1, help="Seconds between samples.")
    parser.add_argument(
        "--unity-snapshot-interval",
        type=float,
        default=0.5,
        help="Seconds between expensive Unity/ptrace body+hammer snapshots. Cursor state still updates every --interval.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=1,
        help="Number of ptrace truth samples used to choose the raw-memory cursor path. Defaults to 1 for safer one-shot reads.",
    )
    parser.add_argument(
        "--calibration-interval",
        type=float,
        default=0.7,
        help="Seconds between calibration samples.",
    )
    parser.add_argument(
        "--window",
        type=lambda value: int(value, 0),
        default=0x200,
        help="Bytes around each root/pointee to scan for matching Vector2 values.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0015,
        help="Float tolerance when matching raw memory to Unity ground truth.",
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
        help="Number of payloads to emit before exiting. Use 0 for an indefinite stream; continuous ptrace streaming is experimental.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="json", help="Output format.")
    args = parser.parse_args()

    pid = args.pid if args.pid is not None else auto_pid()
    if args.samples == 0:
        log("Continuous observation-state streaming uses repeated ptrace remote calls and may destabilize the game process.")
    stream_observation_state(pid, args)


if __name__ == "__main__":
    main()
