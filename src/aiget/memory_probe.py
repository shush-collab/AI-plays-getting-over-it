#!/usr/bin/env python3
import argparse
import os
import struct
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GAME_BINARY = Path("/home/fln/.steam/debian-installation/steamapps/common/Getting Over It/GettingOverIt.x86_64")
GAME_ASSEMBLY = Path("/home/fln/.steam/debian-installation/steamapps/common/Getting Over It/GameAssembly.so")

# These lazy-resolve slots come directly from the current dump.cs / GameAssembly build.
ICALL_SLOTS = {
    "Transform.get_position_Injected": 0x221EE00,
    "Rigidbody2D.get_position_Injected": 0x22251B8,
}


@dataclass(frozen=True)
class MapRegion:
    start: int
    end: int
    perms: str
    offset: int
    path: str

    @property
    def size(self) -> int:
        return self.end - self.start

    def contains(self, addr: int) -> bool:
        return self.start <= addr < self.end


class MemReader:
    def __init__(self, pid: int):
        self.pid = pid
        self.mem_fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
        self.maps = self._load_maps(pid)

    def __enter__(self) -> "MemReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        os.close(self.mem_fd)

    def _load_maps(self, pid: int) -> list[MapRegion]:
        regions: list[MapRegion] = []
        with open(f"/proc/{pid}/maps", "r", encoding="utf-8") as fh:
            for raw in fh:
                parts = raw.split(maxsplit=5)
                if len(parts) < 5:
                    continue
                start_s, end_s = parts[0].split("-")
                path = parts[5].strip() if len(parts) >= 6 else ""
                regions.append(
                    MapRegion(
                        start=int(start_s, 16),
                        end=int(end_s, 16),
                        perms=parts[1],
                        offset=int(parts[2], 16),
                        path=path,
                    )
                )
        return regions

    def region_for(self, addr: int) -> MapRegion | None:
        for region in self.maps:
            if region.contains(addr):
                return region
        return None

    def read(self, addr: int, size: int) -> bytes:
        return os.pread(self.mem_fd, size, addr)

    def read_ptr(self, addr: int) -> int:
        return struct.unpack("<Q", self.read(addr, 8))[0]

    def read_u8(self, addr: int) -> int:
        return self.read(addr, 1)[0]

    def read_u32(self, addr: int) -> int:
        return struct.unpack("<I", self.read(addr, 4))[0]

    def read_i32(self, addr: int) -> int:
        return struct.unpack("<i", self.read(addr, 4))[0]

    def read_f32(self, addr: int) -> float:
        return struct.unpack("<f", self.read(addr, 4))[0]

    def read_vec2(self, addr: int) -> tuple[float, float]:
        return struct.unpack("<ff", self.read(addr, 8))

    def read_vec3(self, addr: int) -> tuple[float, float, float]:
        return struct.unpack("<fff", self.read(addr, 12))


@dataclass
class PlayerControlCandidate:
    addr: int
    klass: int
    native_self: int
    fake_cursor_managed: int
    fake_cursor_klass: int
    fake_cursor_native: int
    fake_cursor_rb_managed: int
    fake_cursor_rb_klass: int
    fake_cursor_rb_native: int
    deadzone: float
    input_enabled: int
    loaded_from_save: int
    load_finished: int
    old_hammer_pos: tuple[float, float, float]
    num_wins: int


def auto_pid() -> int:
    proc = subprocess.run(
        ["pgrep", "-f", str(GAME_BINARY)],
        check=True,
        capture_output=True,
        text=True,
    )
    pids = [int(line) for line in proc.stdout.splitlines() if line.strip()]
    if not pids:
        raise RuntimeError("GettingOverIt.x86_64 is not running")
    return pids[-1]


def fmt_addr(addr: int) -> str:
    return f"0x{addr:016X}"


def describe_addr(reader: MemReader, addr: int) -> str:
    region = reader.region_for(addr)
    if not region:
        return f"{fmt_addr(addr)} [unmapped]"
    rel = addr - region.start + region.offset
    path = region.path or "[anonymous]"
    return f"{fmt_addr(addr)} [{path} + 0x{rel:X}]"


def iter_candidate_regions(reader: MemReader) -> Iterable[MapRegion]:
    for region in reader.maps:
        if "r" not in region.perms or "w" not in region.perms:
            continue
        if "s" in region.perms:
            continue
        if region.size < 0x200 or region.size > 128 * 1024 * 1024:
            continue
        if region.path and region.path != "[heap]":
            continue
        yield region


def is_readable_ptr(reader: MemReader, value: int) -> bool:
    region = reader.region_for(value)
    return bool(region and "r" in region.perms and not (value & 0x7))


def is_rw_private_ptr(reader: MemReader, value: int) -> bool:
    region = reader.region_for(value)
    return bool(region and "r" in region.perms and "w" in region.perms and "s" not in region.perms and not (value & 0x7))


def scan_playercontrol(reader: MemReader, limit: int | None = None) -> list[PlayerControlCandidate]:
    candidates: list[PlayerControlCandidate] = []

    for region in iter_candidate_regions(reader):
        data = reader.read(region.start, region.size)
        for i in range(0, len(data) - 0x118, 8):
            addr = region.start + i
            try:
                klass = struct.unpack_from("<Q", data, i)[0]
                native_self = struct.unpack_from("<Q", data, i + 0x10)[0]
                deadzone = struct.unpack_from("<f", data, i + 0x48)[0]
                fake_cursor_managed = struct.unpack_from("<Q", data, i + 0x50)[0]
                fake_cursor_rb_managed = struct.unpack_from("<Q", data, i + 0x60)[0]
                input_enabled = struct.unpack_from("<B", data, i + 0xA8)[0]
                loaded_from_save = struct.unpack_from("<B", data, i + 0xA9)[0]
                load_finished = struct.unpack_from("<B", data, i + 0xAA)[0]
                old_hammer_pos = struct.unpack_from("<fff", data, i + 0xAC)
                num_wins = struct.unpack_from("<i", data, i + 0x10C)[0]
            except struct.error:
                continue

            if not is_readable_ptr(reader, klass):
                continue
            if not (0.05 < deadzone < 0.5):
                continue
            if input_enabled not in (0, 1) or loaded_from_save not in (0, 1) or load_finished not in (0, 1):
                continue
            if not (0 <= num_wins < 1000):
                continue
            hx, hy, hz = old_hammer_pos
            if not (-20.0 < hx < 80.0 and -5.0 < hy < 200.0 and abs(hz) < 1.0):
                continue
            if not is_rw_private_ptr(reader, fake_cursor_managed) or not is_rw_private_ptr(reader, fake_cursor_rb_managed):
                continue

            fake_cursor_klass = reader.read_ptr(fake_cursor_managed)
            fake_cursor_native = reader.read_ptr(fake_cursor_managed + 0x10)
            fake_cursor_rb_klass = reader.read_ptr(fake_cursor_rb_managed)
            fake_cursor_rb_native = reader.read_ptr(fake_cursor_rb_managed + 0x10)

            if not is_readable_ptr(reader, fake_cursor_klass) or not is_readable_ptr(reader, fake_cursor_rb_klass):
                continue
            if not is_readable_ptr(reader, fake_cursor_native) or not is_readable_ptr(reader, fake_cursor_rb_native):
                continue

            candidates.append(
                PlayerControlCandidate(
                    addr=addr,
                    klass=klass,
                    native_self=native_self,
                    fake_cursor_managed=fake_cursor_managed,
                    fake_cursor_klass=fake_cursor_klass,
                    fake_cursor_native=fake_cursor_native,
                    fake_cursor_rb_managed=fake_cursor_rb_managed,
                    fake_cursor_rb_klass=fake_cursor_rb_klass,
                    fake_cursor_rb_native=fake_cursor_rb_native,
                    deadzone=deadzone,
                    input_enabled=input_enabled,
                    loaded_from_save=loaded_from_save,
                    load_finished=load_finished,
                    old_hammer_pos=old_hammer_pos,
                    num_wins=num_wins,
                )
            )
            if limit is not None and len(candidates) >= limit:
                return candidates
    return candidates


def read_playercontrol_at(reader: MemReader, addr: int) -> PlayerControlCandidate:
    klass = reader.read_ptr(addr)
    native_self = reader.read_ptr(addr + 0x10)
    deadzone = reader.read_f32(addr + 0x48)
    fake_cursor_managed = reader.read_ptr(addr + 0x50)
    fake_cursor_rb_managed = reader.read_ptr(addr + 0x60)
    input_enabled = reader.read_u8(addr + 0xA8)
    loaded_from_save = reader.read_u8(addr + 0xA9)
    load_finished = reader.read_u8(addr + 0xAA)
    old_hammer_pos = reader.read_vec3(addr + 0xAC)
    num_wins = reader.read_i32(addr + 0x10C)

    try:
        fake_cursor_klass = reader.read_ptr(fake_cursor_managed)
        fake_cursor_native = reader.read_ptr(fake_cursor_managed + 0x10)
    except OSError:
        fake_cursor_klass = 0
        fake_cursor_native = 0

    try:
        fake_cursor_rb_klass = reader.read_ptr(fake_cursor_rb_managed)
        fake_cursor_rb_native = reader.read_ptr(fake_cursor_rb_managed + 0x10)
    except OSError:
        fake_cursor_rb_klass = 0
        fake_cursor_rb_native = 0

    return PlayerControlCandidate(
        addr=addr,
        klass=klass,
        native_self=native_self,
        fake_cursor_managed=fake_cursor_managed,
        fake_cursor_klass=fake_cursor_klass,
        fake_cursor_native=fake_cursor_native,
        fake_cursor_rb_managed=fake_cursor_rb_managed,
        fake_cursor_rb_klass=fake_cursor_rb_klass,
        fake_cursor_rb_native=fake_cursor_rb_native,
        deadzone=deadzone,
        input_enabled=input_enabled,
        loaded_from_save=loaded_from_save,
        load_finished=load_finished,
        old_hammer_pos=old_hammer_pos,
        num_wins=num_wins,
    )


def summarize_candidates(candidates: list[PlayerControlCandidate]) -> str:
    if not candidates:
        return "No candidates found."

    klass_counts = Counter(c.klass for c in candidates)
    rb_klass_counts = Counter(c.fake_cursor_rb_klass for c in candidates)
    tf_klass_counts = Counter(c.fake_cursor_klass for c in candidates)

    lines = [
        f"Candidates: {len(candidates)}",
        "Top PlayerControl klass values:",
    ]
    for klass, count in klass_counts.most_common(5):
        lines.append(f"  {fmt_addr(klass)}: {count}")

    lines.append("Top Rigidbody2D wrapper klass values:")
    for klass, count in rb_klass_counts.most_common(5):
        lines.append(f"  {fmt_addr(klass)}: {count}")

    lines.append("Top Transform wrapper klass values:")
    for klass, count in tf_klass_counts.most_common(5):
        lines.append(f"  {fmt_addr(klass)}: {count}")

    dominant_pc_klass = klass_counts.most_common(1)[0][0]
    filtered = [c for c in candidates if c.klass == dominant_pc_klass]
    lines.append("")
    lines.append(f"Likely live PlayerControl wrappers sharing klass {fmt_addr(dominant_pc_klass)}:")
    for candidate in filtered[:10]:
        lines.extend(render_candidate(candidate))
        lines.append("")
    return "\n".join(lines).rstrip()


def render_candidate(candidate: PlayerControlCandidate) -> list[str]:
    hx, hy, hz = candidate.old_hammer_pos
    return [
        f"  PlayerControl @ {fmt_addr(candidate.addr)}",
        f"    native self      {fmt_addr(candidate.native_self)}",
        f"    fakeCursor       {fmt_addr(candidate.fake_cursor_managed)} -> native {fmt_addr(candidate.fake_cursor_native)}",
        f"    fakeCursorRB     {fmt_addr(candidate.fake_cursor_rb_managed)} -> native {fmt_addr(candidate.fake_cursor_rb_native)}",
        f"    deadzone={candidate.deadzone:.4f} input_enabled={candidate.input_enabled} load_finished={candidate.load_finished} wins={candidate.num_wins}",
        f"    oldHammerPos=({hx:.4f}, {hy:.4f}, {hz:.4f})",
    ]


def gameassembly_base(reader: MemReader) -> int:
    for region in reader.maps:
        if region.path == str(GAME_ASSEMBLY) and region.offset == 0:
            return region.start
    raise RuntimeError("GameAssembly.so base mapping not found")


def dump_icalls(reader: MemReader) -> str:
    base = gameassembly_base(reader)
    lines = [f"PID: {reader.pid}", f"GameAssembly base: {fmt_addr(base)}", ""]
    for name, slot in ICALL_SLOTS.items():
        slot_addr = base + slot
        target = reader.read_ptr(slot_addr)
        lines.append(f"{name}")
        lines.append(f"  slot     {describe_addr(reader, slot_addr)}")
        lines.append(f"  target   {describe_addr(reader, target)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def dump_chain(reader: MemReader, addr: int) -> str:
    candidate = read_playercontrol_at(reader, addr)

    lines = [f"PlayerControl chain for {fmt_addr(addr)}", ""]
    lines.append(f"PlayerControl klass:          {describe_addr(reader, candidate.klass)}")
    lines.append(f"PlayerControl native object:  {describe_addr(reader, candidate.native_self)}")
    lines.append(f"fakeCursor wrapper:           {describe_addr(reader, candidate.fake_cursor_managed)}")
    lines.append(f"fakeCursor wrapper klass:     {describe_addr(reader, candidate.fake_cursor_klass)}")
    lines.append(f"fakeCursor native object:     {describe_addr(reader, candidate.fake_cursor_native)}")
    lines.append(f"fakeCursorRB wrapper:         {describe_addr(reader, candidate.fake_cursor_rb_managed)}")
    lines.append(f"fakeCursorRB wrapper klass:   {describe_addr(reader, candidate.fake_cursor_rb_klass)}")
    lines.append(f"fakeCursorRB native object:   {describe_addr(reader, candidate.fake_cursor_rb_native)}")
    return "\n".join(lines)


def dump_region(reader: MemReader, addr: int) -> str:
    region = reader.region_for(addr)
    if not region:
        return f"{fmt_addr(addr)} is unmapped"
    path = region.path or "[anonymous]"
    return (
        f"{fmt_addr(addr)}\n"
        f"  region: {fmt_addr(region.start)}-{fmt_addr(region.end)}\n"
        f"  perms:  {region.perms}\n"
        f"  offset: 0x{region.offset:X}\n"
        f"  path:   {path}\n"
        f"  size:   0x{region.size:X}"
    )


def search_vector2(reader: MemReader, root: int, target_x: float, target_y: float, window: int, eps: float) -> str:
    matches: list[str] = []

    def scan_block(base: int, label: str) -> None:
        region = reader.region_for(base)
        if not region or "r" not in region.perms:
            return
        start = max(region.start, base - window)
        end = min(region.end, base + window)
        data = reader.read(start, end - start)
        for i in range(0, len(data) - 8, 4):
            x, y = struct.unpack_from("<ff", data, i)
            if abs(x - target_x) <= eps and abs(y - target_y) <= eps:
                addr = start + i
                matches.append(f"{label}: {fmt_addr(addr)} -> ({x:.6f}, {y:.6f})")

    scan_block(root, "direct")
    region = reader.region_for(root)
    if region and "r" in region.perms:
        data = reader.read(root, min(0x200, region.end - root))
        for off in range(0, len(data) - 8, 8):
            ptr = struct.unpack_from("<Q", data, off)[0]
            ptr_region = reader.region_for(ptr)
            if not ptr_region or "r" not in ptr_region.perms or "w" not in ptr_region.perms:
                continue
            scan_block(ptr, f"*({fmt_addr(root)} + 0x{off:X})")

    if not matches:
        return "No nearby Vector2 matches found."
    return "\n".join(matches[:80])


def watch_vector2(reader: MemReader, addr: int, interval: float) -> None:
    prev = None
    while True:
        try:
            value = reader.read_vec2(addr)
            rounded = (round(value[0], 6), round(value[1], 6))
            if rounded != prev:
                print(f"{time.time():.3f} {fmt_addr(addr)} x={value[0]:.6f} y={value[1]:.6f}", flush=True)
                prev = rounded
            time.sleep(interval)
        except KeyboardInterrupt:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Getting Over It memory structures on Linux.")
    parser.add_argument("--pid", type=int, default=None, help="Target GettingOverIt PID. Defaults to pgrep.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("icalls", help="Print resolved Unity icall slots for live position getters.")

    scan_parser = subparsers.add_parser("scan-playercontrol", help="Scan for PlayerControl wrappers.")
    scan_parser.add_argument("--limit", type=int, default=None, help="Stop after N validated candidates.")

    chain_parser = subparsers.add_parser("chain", help="Interpret an address as PlayerControl and dump its pointer chain.")
    chain_parser.add_argument("addr", type=lambda value: int(value, 0), help="Suspected PlayerControl address.")

    region_parser = subparsers.add_parser("region", help="Describe the mapped region containing an address.")
    region_parser.add_argument("addr", type=lambda value: int(value, 0), help="Address to resolve.")

    search_parser = subparsers.add_parser("search-vector2", help="Search near an address for a matching Vector2.")
    search_parser.add_argument("root", type=lambda value: int(value, 0), help="Root native address.")
    search_parser.add_argument("x", type=float, help="Target x value.")
    search_parser.add_argument("y", type=float, help="Target y value.")
    search_parser.add_argument("--window", type=lambda value: int(value, 0), default=0x400, help="Bytes around each root/pointee to scan.")
    search_parser.add_argument("--eps", type=float, default=1e-4, help="Absolute float tolerance.")

    watch_parser = subparsers.add_parser("watch-vector2", help="Continuously read a Vector2 from a fixed address.")
    watch_parser.add_argument("addr", type=lambda value: int(value, 0), help="Address of the x float.")
    watch_parser.add_argument("--interval", type=float, default=0.02, help="Seconds between reads.")

    args = parser.parse_args()
    pid = args.pid if args.pid is not None else auto_pid()
    reader = MemReader(pid)
    try:
        if args.cmd == "icalls":
            print(dump_icalls(reader))
        elif args.cmd == "scan-playercontrol":
            print(summarize_candidates(scan_playercontrol(reader, limit=args.limit)))
        elif args.cmd == "chain":
            print(dump_chain(reader, args.addr))
        elif args.cmd == "region":
            print(dump_region(reader, args.addr))
        elif args.cmd == "search-vector2":
            print(search_vector2(reader, args.root, args.x, args.y, args.window, args.eps))
        elif args.cmd == "watch-vector2":
            watch_vector2(reader, args.addr, args.interval)
        else:
            raise AssertionError(f"Unhandled command: {args.cmd}")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
