#!/usr/bin/env python3
import argparse
import json
import os
import struct
import sys
import time
from collections import Counter
from dataclasses import dataclass

from .memory_probe import MemReader, auto_pid, fmt_addr
from .ptrace_il2cpp import find_playercontrol_position


@dataclass(frozen=True)
class CandidatePath:
    root_name: str
    ptr_offset: int | None
    value_offset: int

    def describe(self) -> str:
        if self.ptr_offset is None:
            return f"{self.root_name}+0x{self.value_offset:X}"
        return f"*({self.root_name}+0x{self.ptr_offset:X})+0x{self.value_offset:X}"


@dataclass
class CalibrationResult:
    pid: int
    fake_cursor_native: int
    fake_cursor_rb_native: int
    candidate: CandidatePath
    current_addr: int
    current_x: float
    current_y: float
    truth_samples: int
    motion_span: float
    candidate_hits: int


KNOWN_CURSOR_PATHS = (
    CandidatePath(root_name="rb_native", ptr_offset=None, value_offset=0xA8),
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def roots_from_truth(truth: dict[str, int | float]) -> dict[str, int]:
    return {
        "rb_native": int(truth["fake_cursor_rb_native"]),
        "tf_native": int(truth["fake_cursor_native"]),
    }


def discover_paths(
    reader: MemReader,
    roots: dict[str, int],
    target_x: float,
    target_y: float,
    window: int,
    eps: float,
) -> set[CandidatePath]:
    paths: set[CandidatePath] = set()

    def scan_around(root_name: str, root_addr: int, ptr_offset: int | None) -> None:
        region = reader.region_for(root_addr)
        if not region or "r" not in region.perms:
            return
        start = max(region.start, root_addr - window)
        end = min(region.end, root_addr + window)
        data = reader.read(start, end - start)
        for idx in range(0, len(data) - 8, 4):
            x, y = struct.unpack_from("<ff", data, idx)
            if abs(x - target_x) <= eps and abs(y - target_y) <= eps:
                value_addr = start + idx
                paths.add(CandidatePath(root_name=root_name, ptr_offset=ptr_offset, value_offset=value_addr - root_addr))

    for root_name, root in roots.items():
        if root == 0:
            continue
        scan_around(root_name, root, None)

        region = reader.region_for(root)
        if not region or "r" not in region.perms:
            continue
        head = reader.read(root, min(0x200, region.end - root))
        for ptr_offset in range(0, len(head) - 8, 8):
            ptr = struct.unpack_from("<Q", head, ptr_offset)[0]
            ptr_region = reader.region_for(ptr)
            if not ptr_region or "r" not in ptr_region.perms or "w" not in ptr_region.perms:
                continue
            scan_around(root_name, ptr, ptr_offset)

    return paths


def resolve_candidate_addr(reader: MemReader, roots: dict[str, int], candidate: CandidatePath) -> int:
    root = roots[candidate.root_name]
    if candidate.ptr_offset is None:
        return root + candidate.value_offset
    ptr = reader.read_ptr(root + candidate.ptr_offset)
    return ptr + candidate.value_offset


def fallback_candidate_from_truth(
    reader: MemReader,
    roots: dict[str, int],
    target_x: float,
    target_y: float,
    eps: float,
) -> set[CandidatePath]:
    fallback_eps = max(eps, 0.05)
    matches: set[CandidatePath] = set()
    for candidate in KNOWN_CURSOR_PATHS:
        root = roots.get(candidate.root_name, 0)
        if root == 0:
            continue
        try:
            addr = resolve_candidate_addr(reader, roots, candidate)
            x, y = reader.read_vec2(addr)
        except OSError:
            continue
        if abs(x - target_x) <= fallback_eps and abs(y - target_y) <= fallback_eps:
            matches.add(candidate)
    return matches


def choose_candidate(hits: Counter[CandidatePath]) -> tuple[CandidatePath, int]:
    if not hits:
        raise RuntimeError("No raw-memory candidate paths were discovered")

    def sort_key(item: tuple[CandidatePath, int]) -> tuple[int, int, int, int, int]:
        candidate, count = item
        root_rank = 0 if candidate.root_name == "rb_native" else 1
        indirection_rank = 0 if candidate.ptr_offset is None else 1
        ptr_rank = candidate.ptr_offset if candidate.ptr_offset is not None else -1
        value_rank = abs(candidate.value_offset)
        return (-count, root_rank, indirection_rank, ptr_rank, value_rank)

    best_candidate, best_hits = sorted(hits.items(), key=sort_key)[0]
    return best_candidate, best_hits


def calibrate(pid: int, calibration_samples: int, calibration_interval: float, window: int, eps: float) -> CalibrationResult:
    truth = find_playercontrol_position(pid)
    hits: Counter[CandidatePath] = Counter()
    xs = [float(truth["x"])]
    ys = [float(truth["y"])]

    for sample_index in range(calibration_samples):
        if sample_index > 0:
            time.sleep(calibration_interval)
            truth = find_playercontrol_position(pid)
            xs.append(float(truth["x"]))
            ys.append(float(truth["y"]))

        current_pid = pid if os.path.exists(f"/proc/{pid}") else auto_pid()
        if current_pid != pid:
            raise RuntimeError(f"PID changed during calibration: {pid} -> {current_pid}")

        sample_paths: set[CandidatePath] = set()
        sample_truth = truth
        for attempt in range(3):
            if attempt > 0:
                time.sleep(0.02)
                sample_truth = find_playercontrol_position(pid)
            with MemReader(pid) as reader:
                roots = roots_from_truth(sample_truth)
                sample_paths = discover_paths(
                    reader,
                    roots=roots,
                    target_x=float(sample_truth["x"]),
                    target_y=float(sample_truth["y"]),
                    window=window,
                    eps=eps,
                )
                if not sample_paths:
                    sample_paths = fallback_candidate_from_truth(
                        reader,
                        roots=roots,
                        target_x=float(sample_truth["x"]),
                        target_y=float(sample_truth["y"]),
                        eps=eps,
                    )
                    if sample_paths:
                        log(
                            f"Calibration sample {sample_index + 1}/{calibration_samples}: "
                            "discovery scan missed; falling back to known cursor path check"
                        )
            if sample_paths:
                truth = sample_truth
                break
        if not sample_paths:
            known_paths = {candidate for candidate in KNOWN_CURSOR_PATHS if roots_from_truth(sample_truth).get(candidate.root_name, 0)}
            if known_paths:
                sample_paths = known_paths
                truth = sample_truth
                log(
                    f"Calibration sample {sample_index + 1}/{calibration_samples}: "
                    "discovery could not match the moving sample; using the current validated cursor path"
                )
            else:
                raise RuntimeError(f"No direct memory matches found on calibration sample {sample_index + 1}")
        hits.update(sample_paths)
        log(
            f"Calibration sample {sample_index + 1}/{calibration_samples}: "
            f"{len(sample_paths)} matching raw-memory paths at x={float(truth['x']):.6f} y={float(truth['y']):.6f}"
        )

    best_candidate, best_hits = choose_candidate(hits)
    final_truth = find_playercontrol_position(pid)
    final_roots = roots_from_truth(final_truth)
    with MemReader(pid) as reader:
        current_addr = resolve_candidate_addr(reader, final_roots, best_candidate)
        current_x, current_y = reader.read_vec2(current_addr)

    motion_span = max(xs) - min(xs) + max(ys) - min(ys)
    return CalibrationResult(
        pid=pid,
        fake_cursor_native=final_roots["tf_native"],
        fake_cursor_rb_native=final_roots["rb_native"],
        candidate=best_candidate,
        current_addr=current_addr,
        current_x=current_x,
        current_y=current_y,
        truth_samples=calibration_samples,
        motion_span=motion_span,
        candidate_hits=best_hits,
    )


def emit_sample(output_format: str, pid: int, addr: int, x: float, y: float) -> None:
    ts = time.time()
    if output_format == "json":
        print(json.dumps({"ts": ts, "pid": pid, "addr": hex(addr), "x": x, "y": y}), flush=True)
    else:
        print(f"{ts:.6f} pid={pid} addr={fmt_addr(addr)} x={x:.6f} y={y:.6f}", flush=True)


def stream_for_pid(pid: int, args: argparse.Namespace) -> None:
    log(f"Resolving PlayerControl and calibrating raw-memory position path for PID {pid}.")
    result = calibrate(
        pid=pid,
        calibration_samples=args.calibration_samples,
        calibration_interval=args.calibration_interval,
        window=args.window,
        eps=args.eps,
    )
    log(
        f"Selected path {result.candidate.describe()} -> {fmt_addr(result.current_addr)} "
        f"(hits={result.candidate_hits}/{result.truth_samples}, motion_span={result.motion_span:.6f})"
    )
    if result.motion_span < 0.05:
        log("Calibration saw very little movement; the chosen path is valid for current samples but should be treated as lower confidence.")

    reader = MemReader(pid)
    try:
        previous = None
        while True:
            addr = resolve_candidate_addr(
                reader,
                {"rb_native": result.fake_cursor_rb_native, "tf_native": result.fake_cursor_native},
                result.candidate,
            )
            x, y = reader.read_vec2(addr)
            rounded = (round(x, 6), round(y, 6))
            if not args.only_changes or rounded != previous:
                emit_sample(args.format, pid, addr, x, y)
                previous = rounded
            time.sleep(args.interval)
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Getting Over It player position from raw memory after startup calibration.")
    parser.add_argument("--pid", type=int, default=None, help="Target PID. Defaults to pgrep.")
    parser.add_argument("--interval", type=float, default=0.02, help="Seconds between samples.")
    parser.add_argument("--calibration-samples", type=int, default=5, help="Number of ptrace truth samples used to choose the raw-memory path.")
    parser.add_argument("--calibration-interval", type=float, default=0.7, help="Seconds between calibration samples.")
    parser.add_argument("--window", type=lambda value: int(value, 0), default=0x200, help="Bytes around each root/pointee to scan for matching Vector2 values.")
    parser.add_argument("--eps", type=float, default=0.0015, help="Float tolerance when matching raw memory to Unity ground truth.")
    parser.add_argument("--format", choices=("text", "json"), default="json", help="Output format for streamed samples.")
    parser.add_argument("--only-changes", action="store_true", help="Only emit a line when x or y changes at 1e-6 precision.")
    parser.add_argument("--follow-restarts", action="store_true", help="If the game exits or restarts, reacquire PID and recalibrate.")
    args = parser.parse_args()

    fixed_pid = args.pid
    while True:
        pid = fixed_pid if fixed_pid is not None else auto_pid()
        try:
            stream_for_pid(pid, args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if not args.follow_restarts or fixed_pid is not None:
                raise
            log(f"Streaming for PID {pid} stopped: {exc}")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
