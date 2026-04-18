#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import dataclass

from .live_position import calibrate, log, resolve_candidate_addr
from .memory_probe import MemReader, auto_pid, fmt_addr

@dataclass(frozen=True)
class PositionSample:
    ts: float
    x: float
    y: float


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


def estimate_velocity(previous: PositionSample | None, current: PositionSample) -> tuple[float, float]:
    if previous is None:
        return 0.0, 0.0
    dt = current.ts - previous.ts
    if dt <= 0:
        return 0.0, 0.0
    return (current.x - previous.x) / dt, (current.y - previous.y) / dt


def format_payload(
    *,
    ts: float,
    pid: int,
    addr: int,
    x: float,
    y: float,
    vx: float,
    vy: float,
    progress: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "ts": ts,
        "pid": pid,
        "addr": hex(addr),
        "implemented_features": [
            "cursor_position_xy",
            "cursor_velocity_xy",
            "progress_features",
        ],
        "cursor_position_xy": [x, y],
        "cursor_velocity_xy": [vx, vy],
        "progress_features": list(progress),
    }


def stream_observation_state(pid: int, args: argparse.Namespace) -> None:
    log(f"Resolving PlayerControl and calibrating observation state path for PID {pid}.")
    calibration = calibrate(
        pid=pid,
        calibration_samples=args.calibration_samples,
        calibration_interval=args.calibration_interval,
        window=args.window,
        eps=args.eps,
    )
    log(
        f"Selected cursor path {calibration.candidate.describe()} -> {fmt_addr(calibration.current_addr)} "
        f"(hits={calibration.candidate_hits}/{calibration.truth_samples}, motion_span={calibration.motion_span:.6f})"
    )

    tracker = ProgressTracker(best_height=calibration.current_y, last_progress_ts=time.time())
    previous: PositionSample | None = None
    reader = MemReader(pid)
    try:
        while True:
            now = time.time()
            addr = resolve_candidate_addr(
                reader,
                {
                    "rb_native": calibration.fake_cursor_rb_native,
                    "tf_native": calibration.fake_cursor_native,
                },
                calibration.candidate,
            )
            x, y = reader.read_vec2(addr)
            current = PositionSample(ts=now, x=x, y=y)
            vx, vy = estimate_velocity(previous, current)
            progress = tracker.update(current_height=y, now=now)
            payload = format_payload(
                ts=now,
                pid=pid,
                addr=addr,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                progress=progress,
            )
            if args.format == "json":
                print(json.dumps(payload), flush=True)
            else:
                print(
                    f"{payload['ts']:.6f} pid={pid} addr={fmt_addr(addr)} "
                    f"cursor=({x:.6f}, {y:.6f}) velocity=({vx:.6f}, {vy:.6f}) "
                    f"progress=({progress[0]:.6f}, {progress[1]:.6f}, {progress[2]:.6f})",
                    flush=True,
                )
            previous = current
            time.sleep(args.interval)
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream the currently implemented observation-state features for Getting Over It."
    )
    parser.add_argument("--pid", type=int, default=None, help="Target PID. Defaults to pgrep.")
    parser.add_argument("--interval", type=float, default=0.05, help="Seconds between samples.")
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=5,
        help="Number of ptrace truth samples used to choose the raw-memory cursor path.",
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
    parser.add_argument("--format", choices=("text", "json"), default="json", help="Output format.")
    args = parser.parse_args()

    pid = args.pid if args.pid is not None else auto_pid()
    stream_observation_state(pid, args)


if __name__ == "__main__":
    main()
