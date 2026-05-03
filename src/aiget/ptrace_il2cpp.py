#!/usr/bin/env python3
import argparse
import ctypes
import ctypes.util
import math
import os
import signal
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .live_layout import ResolvedLiveLayout
from .memory_probe import MemReader, read_optional_f32, read_optional_vec2
from .memory_probe import auto_pid


GAME_ASSEMBLY = Path("/home/fln/.steam/debian-installation/steamapps/common/Getting Over It/GameAssembly.so")
ICALL_RIGIDBODY2D_GET_POSITION = "UnityEngine.Rigidbody2D::get_position_Injected(UnityEngine.Vector2&)"
ICALL_TRANSFORM_GET_POSITION = "UnityEngine.Transform::get_position_Injected(UnityEngine.Vector3&)"

PTRACE_ATTACH = 16
PTRACE_DETACH = 17
PTRACE_CONT = 7
PTRACE_GETREGS = 12
PTRACE_SETREGS = 13
PTRACE_PEEKDATA = 2
PTRACE_POKEDATA = 5
PTRACE_GETREGSET = 0x4204
PTRACE_SETREGSET = 0x4205

NT_PRFPREG = 2
NT_X86_XSTATE = 0x202

REGSET_BUFFER_SIZE = 4096
TRAP_SEARCH_RADIUS = 0x100
_EXPORT_OFFSETS_CACHE: dict[str, int] | None = None


class user_regs_struct(ctypes.Structure):
    _fields_ = [
        ("r15", ctypes.c_ulonglong),
        ("r14", ctypes.c_ulonglong),
        ("r13", ctypes.c_ulonglong),
        ("r12", ctypes.c_ulonglong),
        ("rbp", ctypes.c_ulonglong),
        ("rbx", ctypes.c_ulonglong),
        ("r11", ctypes.c_ulonglong),
        ("r10", ctypes.c_ulonglong),
        ("r9", ctypes.c_ulonglong),
        ("r8", ctypes.c_ulonglong),
        ("rax", ctypes.c_ulonglong),
        ("rcx", ctypes.c_ulonglong),
        ("rdx", ctypes.c_ulonglong),
        ("rsi", ctypes.c_ulonglong),
        ("rdi", ctypes.c_ulonglong),
        ("orig_rax", ctypes.c_ulonglong),
        ("rip", ctypes.c_ulonglong),
        ("cs", ctypes.c_ulonglong),
        ("eflags", ctypes.c_ulonglong),
        ("rsp", ctypes.c_ulonglong),
        ("ss", ctypes.c_ulonglong),
        ("fs_base", ctypes.c_ulonglong),
        ("gs_base", ctypes.c_ulonglong),
        ("ds", ctypes.c_ulonglong),
        ("es", ctypes.c_ulonglong),
        ("fs", ctypes.c_ulonglong),
        ("gs", ctypes.c_ulonglong),
    ]


class iovec(ctypes.Structure):
    _fields_ = [
        ("iov_base", ctypes.c_void_p),
        ("iov_len", ctypes.c_size_t),
    ]


@dataclass(frozen=True)
class MapRegion:
    start: int
    end: int
    perms: str
    offset: int
    path: str

    def contains(self, addr: int) -> bool:
        return self.start <= addr < self.end


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    mismatches: dict[str, float]
    notes: list[str]


@dataclass(frozen=True)
class _PlayerControlRefs:
    player_obj: int
    fake_cursor_obj: int
    fake_cursor_native: int
    fake_cursor_rb_obj: int
    fake_cursor_rb_native: int
    body_transform_obj: int
    body_transform_native: int
    hammer_anchor_transform_obj: int
    hammer_anchor_transform_native: int
    hammer_tip_transform_obj: int
    hammer_tip_transform_native: int
    hammer_collisions_obj: int


@dataclass(frozen=True)
class _ObservationSnapshot:
    body_position_xy: tuple[float, float]
    body_angle: float
    hammer_anchor_xy: tuple[float, float]
    hammer_tip_xy: tuple[float, float]
    hammer_contact_flags: tuple[float, float]
    hammer_contact_normal_xy: tuple[float, float]


class RemoteError(RuntimeError):
    pass


libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.ptrace.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
libc.ptrace.restype = ctypes.c_long


def ptrace(request: int, pid: int, addr: int = 0, data: int = 0) -> int:
    result = libc.ptrace(request, pid, ctypes.c_void_p(addr), ctypes.c_void_p(data))
    if result == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return result


def load_maps(pid: int) -> list[MapRegion]:
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


def region_for(maps: list[MapRegion], addr: int) -> MapRegion | None:
    for region in maps:
        if region.contains(addr):
            return region
    return None


def gameassembly_base(maps: list[MapRegion]) -> int:
    for region in maps:
        if region.path == str(GAME_ASSEMBLY) and region.offset == 0:
            return region.start
    raise RemoteError("GameAssembly base not found")


def load_export_offsets() -> dict[str, int]:
    global _EXPORT_OFFSETS_CACHE
    if _EXPORT_OFFSETS_CACHE is not None:
        return dict(_EXPORT_OFFSETS_CACHE)

    wanted = {
        "il2cpp_assembly_get_image",
        "il2cpp_class_from_name",
        "il2cpp_class_get_field_from_name",
        "il2cpp_class_get_method_from_name",
        "il2cpp_class_get_type",
        "il2cpp_domain_get",
        "il2cpp_domain_get_assemblies",
        "il2cpp_field_get_offset",
        "il2cpp_field_get_value",
        "il2cpp_image_get_name",
        "il2cpp_resolve_icall",
        "il2cpp_runtime_invoke",
        "il2cpp_thread_attach",
        "il2cpp_type_get_object",
    }
    proc = subprocess.run(
        ["nm", "-D", str(GAME_ASSEMBLY)],
        check=True,
        capture_output=True,
        text=True,
    )
    offsets: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[1] != "T":
            continue
        name = parts[2]
        if name in wanted:
            offsets[name] = int(parts[0], 16)
    missing = sorted(wanted - offsets.keys())
    if missing:
        raise RemoteError(f"Missing GameAssembly exports: {', '.join(missing)}")
    _EXPORT_OFFSETS_CACHE = dict(offsets)
    return dict(offsets)


def choose_int3_trap_offset(data: bytes, pivot: int) -> int | None:
    if not data:
        return None

    pivot = max(0, min(pivot, len(data) - 1))
    best_idx = None
    best_key = None

    for idx, value in enumerate(data):
        if value != 0xCC:
            continue
        key = (abs(idx - pivot), 0 if idx <= pivot else 1, idx)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = idx

    return best_idx


class RemoteProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.maps = load_maps(pid)
        self.threads = sorted(int(tid) for tid in os.listdir(f"/proc/{pid}/task"))
        if pid not in self.threads:
            self.threads.insert(0, pid)
        self.leader = pid
        self.mem_fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
        self._attached = False
        self._base_regs: user_regs_struct | None = None
        self._base_regset_note: int | None = None
        self._base_regset = b""
        self._scratch_base = 0
        self._scratch_size = 0x4000
        self._scratch_top = 0
        self._scratch_orig = b""
        self._trap_addr = 0

    def close(self) -> None:
        if self._attached:
            self.detach_all()
        os.close(self.mem_fd)

    def read(self, addr: int, size: int) -> bytes:
        return os.pread(self.mem_fd, size, addr)

    def read_ptr(self, addr: int) -> int:
        return struct.unpack("<Q", self.read(addr, 8))[0]

    def read_cstring(self, addr: int, limit: int = 256) -> str:
        data = bytearray()
        while len(data) < limit:
            chunk = self.read(addr + len(data), 32)
            if not chunk:
                break
            if b"\x00" in chunk:
                data.extend(chunk.split(b"\x00", 1)[0])
                break
            data.extend(chunk)
        return data.decode("utf-8", errors="replace")

    def write(self, addr: int, data: bytes) -> None:
        padded = data + b"\x00" * ((8 - (len(data) % 8)) % 8)
        for offset in range(0, len(padded), 8):
            word = struct.unpack("<Q", padded[offset : offset + 8])[0]
            ptrace(PTRACE_POKEDATA, self.leader, addr + offset, word)

    def get_regs(self) -> user_regs_struct:
        regs = user_regs_struct()
        ptrace(PTRACE_GETREGS, self.leader, 0, ctypes.addressof(regs))
        return regs

    def set_regs(self, regs: user_regs_struct) -> None:
        ptrace(PTRACE_SETREGS, self.leader, 0, ctypes.addressof(regs))

    def _get_regset(self, note_type: int, size: int = REGSET_BUFFER_SIZE) -> bytes:
        buffer = ctypes.create_string_buffer(size)
        iov = iovec(iov_base=ctypes.cast(buffer, ctypes.c_void_p), iov_len=size)
        ptrace(PTRACE_GETREGSET, self.leader, note_type, ctypes.addressof(iov))
        actual_size = int(iov.iov_len)
        if actual_size <= 0:
            raise RemoteError(f"PTRACE_GETREGSET returned empty state for note type 0x{note_type:X}")
        return buffer.raw[:actual_size]

    def _set_regset(self, note_type: int, data: bytes) -> None:
        buffer = ctypes.create_string_buffer(data)
        iov = iovec(iov_base=ctypes.cast(buffer, ctypes.c_void_p), iov_len=len(data))
        ptrace(PTRACE_SETREGSET, self.leader, note_type, ctypes.addressof(iov))

    def _capture_extended_state(self) -> tuple[int, bytes]:
        errors: list[str] = []
        for note_type in (NT_X86_XSTATE, NT_PRFPREG):
            try:
                return note_type, self._get_regset(note_type)
            except OSError as exc:
                errors.append(f"0x{note_type:X}: {exc}")
        raise RemoteError(
            "Unable to capture floating-point/SIMD register state before remote call: " + "; ".join(errors)
        )

    def _restore_extended_state(self) -> None:
        if self._base_regset_note is None or not self._base_regset:
            return
        self._set_regset(self._base_regset_note, self._base_regset)

    def _resolve_trap_addr(self) -> int:
        # Resolve the current Transform.get_position icall target and search nearby executable bytes
        # for an existing int3 instruction we can safely use as a synthetic return breakpoint.
        game_base = gameassembly_base(self.maps)
        transform_slot = game_base + 0x221EE00
        transform_target = self.read_ptr(transform_slot)
        if not transform_target:
            raise RemoteError("Transform.get_position icall slot resolved to null")

        region = region_for(self.maps, transform_target)
        if not region or "x" not in region.perms:
            raise RemoteError(
                f"Transform.get_position target 0x{transform_target:X} is not in an executable mapping"
            )

        window_start = max(region.start, transform_target - TRAP_SEARCH_RADIUS)
        window_end = min(region.end, transform_target + TRAP_SEARCH_RADIUS)
        code = self.read(window_start, window_end - window_start)
        trap_offset = choose_int3_trap_offset(code, transform_target - window_start)
        if trap_offset is None:
            raise RemoteError(
                "No nearby int3 breakpoint was found around Transform.get_position; refusing remote execution"
            )
        return window_start + trap_offset

    def attach_all(self) -> None:
        if self._attached:
            return
        for tid in self.threads:
            ptrace(PTRACE_ATTACH, tid)
        for tid in self.threads:
            os.waitpid(tid, 0)
        self._attached = True
        self._base_regs = self.get_regs()
        self._base_regset_note, self._base_regset = self._capture_extended_state()
        self._trap_addr = self._resolve_trap_addr()

        base_regs = self._base_regs
        self._scratch_top = (base_regs.rsp - 0x2000) & ~0xF
        self._scratch_base = self._scratch_top - self._scratch_size
        self._scratch_orig = self.read(self._scratch_base, self._scratch_size)

    def detach_all(self) -> None:
        if not self._attached:
            return
        if self._base_regs is not None:
            self.write(self._scratch_base, self._scratch_orig)
            self.set_regs(self._base_regs)
            self._restore_extended_state()
        for tid in reversed(self.threads):
            ptrace(PTRACE_DETACH, tid, 0, 0)
        self._attached = False

    def reset_call_frame(self) -> None:
        assert self._base_regs is not None
        self.write(self._scratch_base, self._scratch_orig)
        self.set_regs(self._base_regs)
        self._restore_extended_state()
        self._cursor = self._scratch_base

    def alloc(self, data: bytes, align: int = 8) -> int:
        cursor = (self._cursor + (align - 1)) & ~(align - 1)
        if cursor + len(data) >= self._scratch_top - 0x100:
            raise RemoteError("Scratch stack exhausted")
        self.write(cursor, data)
        self._cursor = cursor + len(data)
        return cursor

    def alloc_cstring(self, text: str) -> int:
        return self.alloc(text.encode("utf-8") + b"\x00", align=8)

    def execute(self, func_addr: int, args: list[int]) -> user_regs_struct:
        if len(args) > 6:
            raise RemoteError("Only up to 6 register arguments are supported")
        assert self._base_regs is not None
        regs = user_regs_struct.from_buffer_copy(bytes(self._base_regs))

        entry_rsp = self._scratch_top - 8
        self.write(entry_rsp, struct.pack("<Q", self._trap_addr))
        regs.rsp = entry_rsp
        regs.rip = func_addr
        regs.rax = 0

        reg_names = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
        for name, value in zip(reg_names, args):
            setattr(regs, name, value)

        self.set_regs(regs)
        ptrace(PTRACE_CONT, self.leader, 0, 0)
        _, status = os.waitpid(self.leader, 0)
        if os.WIFSTOPPED(status):
            sig = os.WSTOPSIG(status)
            if sig != signal.SIGTRAP:
                raise RemoteError(f"Remote call stopped with signal {sig}")
        else:
            raise RemoteError(f"Unexpected wait status: 0x{status:X}")
        return self.get_regs()


class Il2CppRemote:
    def __init__(self, process: RemoteProcess):
        self.process = process
        self.export_offsets = load_export_offsets()
        self.base = gameassembly_base(process.maps)
        self.exports = {name: self.base + offset for name, offset in self.export_offsets.items()}
        self._icall_cache: dict[str, int] = {}

    def call(self, name: str, args: list[int]) -> int:
        self.process.reset_call_frame()
        regs = self.process.execute(self.exports[name], args)
        return regs.rax

    def call_with_setup(self, name: str, args: list[int], readbacks: list[tuple[int, int]]) -> tuple[int, list[bytes]]:
        self.process.reset_call_frame()
        regs = self.process.execute(self.exports[name], args)
        values = [self.process.read(addr, size) for addr, size in readbacks]
        return regs.rax, values

    def find_images(self) -> dict[str, int]:
        self.process.reset_call_frame()
        count_ptr = self.process.alloc(struct.pack("<Q", 0))
        regs = self.process.execute(self.exports["il2cpp_domain_get"], [])
        domain = regs.rax
        self.process.reset_call_frame()
        count_ptr = self.process.alloc(struct.pack("<Q", 0))
        regs = self.process.execute(self.exports["il2cpp_domain_get_assemblies"], [domain, count_ptr])
        assemblies_ptr = regs.rax
        count = struct.unpack("<Q", self.process.read(count_ptr, 8))[0]
        images: dict[str, int] = {}
        for index in range(count):
            assembly = self.process.read_ptr(assemblies_ptr + index * 8)
            image = self.call("il2cpp_assembly_get_image", [assembly])
            name_ptr = self.call("il2cpp_image_get_name", [image])
            name = self.process.read_cstring(name_ptr, limit=256)
            images[name] = image
        return images

    def class_from_name(self, image: int, namespace: str, name: str) -> int:
        self.process.reset_call_frame()
        ns_ptr = self.process.alloc_cstring(namespace)
        name_ptr = self.process.alloc_cstring(name)
        regs = self.process.execute(self.exports["il2cpp_class_from_name"], [image, ns_ptr, name_ptr])
        return regs.rax

    def method_from_name(self, klass: int, name: str, argc: int) -> int:
        self.process.reset_call_frame()
        name_ptr = self.process.alloc_cstring(name)
        regs = self.process.execute(self.exports["il2cpp_class_get_method_from_name"], [klass, name_ptr, argc])
        return regs.rax

    def field_from_name(self, klass: int, name: str) -> int:
        self.process.reset_call_frame()
        name_ptr = self.process.alloc_cstring(name)
        regs = self.process.execute(self.exports["il2cpp_class_get_field_from_name"], [klass, name_ptr])
        return regs.rax

    def resolve_icall(self, name: str) -> int:
        cached = self._icall_cache.get(name)
        if cached is not None:
            return cached
        self.process.reset_call_frame()
        name_ptr = self.process.alloc_cstring(name)
        regs = self.process.execute(self.exports["il2cpp_resolve_icall"], [name_ptr])
        if not regs.rax:
            raise RemoteError(f"il2cpp_resolve_icall returned null for {name}")
        self._icall_cache[name] = regs.rax
        return regs.rax

    def invoke(self, method: int, obj: int, params: list[int]) -> int:
        self.process.reset_call_frame()
        exc_ptr = self.process.alloc(struct.pack("<Q", 0))
        params_ptr = self.process.alloc(struct.pack("<" + "Q" * len(params), *params)) if params else 0
        regs = self.process.execute(self.exports["il2cpp_runtime_invoke"], [method, obj, params_ptr, exc_ptr])
        exc = self.process.read_ptr(exc_ptr)
        if exc:
            raise RemoteError(f"il2cpp_runtime_invoke raised exception object at 0x{exc:X}")
        return regs.rax

    def get_field_value_ptr(self, obj: int, field: int) -> int:
        self.process.reset_call_frame()
        out_ptr = self.process.alloc(struct.pack("<Q", 0))
        self.process.execute(self.exports["il2cpp_field_get_value"], [obj, field, out_ptr])
        return self.process.read_ptr(out_ptr)

    def get_position_via_icall(self, rb_obj: int) -> tuple[float, float]:
        fn = self.resolve_icall(ICALL_RIGIDBODY2D_GET_POSITION)
        self.process.reset_call_frame()
        out_ptr = self.process.alloc(b"\x00" * 8)
        self.process.execute(fn, [rb_obj, out_ptr])
        return struct.unpack("<ff", self.process.read(out_ptr, 8))


def _require_object_field(runtime: Il2CppRemote, klass: int, obj: int, field_name: str, owner_name: str) -> int:
    field = runtime.field_from_name(klass, field_name)
    if not field:
        raise RemoteError(f"{owner_name}.{field_name} field lookup failed")
    value = runtime.get_field_value_ptr(obj, field)
    if not value:
        raise RemoteError(f"{owner_name}.{field_name} resolved to null")
    return value


def _wrap_angle_delta(delta: float) -> float:
    while delta <= -math.pi:
        delta += 2.0 * math.pi
    while delta > math.pi:
        delta -= 2.0 * math.pi
    return delta


def _decode_hammer_contact_state(
    *, slide: bool, contact_count: int, first_contact: bytes | None
) -> tuple[tuple[float, float], tuple[float, float]]:
    if contact_count <= 0 or first_contact is None:
        return (0.0, 1.0 if slide else 0.0), (0.0, 0.0)

    _, _, normal_x, normal_y = struct.unpack("<ffff", first_contact[:16])
    return (1.0, 1.0 if slide else 0.0), (normal_x, normal_y)


def _quaternion_to_z_angle(quaternion: tuple[float, float, float, float]) -> float:
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _attach_runtime(pid: int) -> tuple[RemoteProcess, Il2CppRemote]:
    process = RemoteProcess(pid)
    try:
        process.attach_all()
        runtime = Il2CppRemote(process)
        domain = runtime.call("il2cpp_domain_get", [])
        runtime.call("il2cpp_thread_attach", [domain])
        return process, runtime
    except Exception:
        process.close()
        raise


def _resolve_playercontrol_refs(runtime: Il2CppRemote, process: RemoteProcess) -> _PlayerControlRefs:
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
    hammer_collisions_obj = _require_object_field(runtime, player_class, player_obj, "hammerCollisions", "PlayerControl")

    body_transform_obj = _require_object_field(runtime, pose_class, pose_obj, "potMeshHub", "PoseControl")
    hammer_anchor_transform_obj = _require_object_field(runtime, pose_class, pose_obj, "handle", "PoseControl")
    hammer_tip_transform_obj = _require_object_field(runtime, pose_class, pose_obj, "tip", "PoseControl")

    return _PlayerControlRefs(
        player_obj=player_obj,
        fake_cursor_obj=fake_cursor_obj,
        fake_cursor_native=process.read_ptr(fake_cursor_obj + 0x10),
        fake_cursor_rb_obj=fake_cursor_rb_obj,
        fake_cursor_rb_native=process.read_ptr(fake_cursor_rb_obj + 0x10),
        body_transform_obj=body_transform_obj,
        body_transform_native=process.read_ptr(body_transform_obj + 0x10),
        hammer_anchor_transform_obj=hammer_anchor_transform_obj,
        hammer_anchor_transform_native=process.read_ptr(hammer_anchor_transform_obj + 0x10),
        hammer_tip_transform_obj=hammer_tip_transform_obj,
        hammer_tip_transform_native=process.read_ptr(hammer_tip_transform_obj + 0x10),
        hammer_collisions_obj=hammer_collisions_obj,
    )


def _sample_transform_position(runtime: Il2CppRemote, transform_obj: int) -> tuple[float, float, float]:
    fn = runtime.resolve_icall(ICALL_TRANSFORM_GET_POSITION)
    runtime.process.reset_call_frame()
    out_ptr = runtime.process.alloc(b"\x00" * 12)
    runtime.process.execute(fn, [transform_obj, out_ptr])
    return struct.unpack("<fff", runtime.process.read(out_ptr, 12))


def _sample_transform_rotation(runtime: Il2CppRemote, transform_obj: int) -> tuple[float, float, float, float]:
    fn = runtime.resolve_icall("UnityEngine.Transform::get_rotation_Injected(UnityEngine.Quaternion&)")
    runtime.process.reset_call_frame()
    out_ptr = runtime.process.alloc(b"\x00" * 16)
    runtime.process.execute(fn, [transform_obj, out_ptr])
    return struct.unpack("<ffff", runtime.process.read(out_ptr, 16))


def _sample_hammer_contact_state(
    process: RemoteProcess,
    hammer_collisions_obj: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    slide = bool(process.read(hammer_collisions_obj + 0x4C, 1)[0])
    contacts_obj = process.read_ptr(hammer_collisions_obj + 0x268)
    if not contacts_obj:
        return _decode_hammer_contact_state(slide=slide, contact_count=0, first_contact=None)

    contact_count = int(process.read_ptr(contacts_obj + 0x18))
    first_contact = None
    if contact_count > 0:
        first_contact = process.read(contacts_obj + 0x20, 16)
    return _decode_hammer_contact_state(slide=slide, contact_count=contact_count, first_contact=first_contact)


def _sample_observation_snapshot(
    runtime: Il2CppRemote,
    process: RemoteProcess,
    refs: _PlayerControlRefs,
) -> _ObservationSnapshot:
    body_position = _sample_transform_position(runtime, refs.body_transform_obj)
    hammer_anchor = _sample_transform_position(runtime, refs.hammer_anchor_transform_obj)
    hammer_tip = _sample_transform_position(runtime, refs.hammer_tip_transform_obj)
    body_rotation = _sample_transform_rotation(runtime, refs.body_transform_obj)
    hammer_contact_flags, hammer_contact_normal = _sample_hammer_contact_state(process, refs.hammer_collisions_obj)

    return _ObservationSnapshot(
        body_position_xy=(body_position[0], body_position[1]),
        body_angle=_quaternion_to_z_angle(body_rotation),
        hammer_anchor_xy=(hammer_anchor[0], hammer_anchor[1]),
        hammer_tip_xy=(hammer_tip[0], hammer_tip[1]),
        hammer_contact_flags=hammer_contact_flags,
        hammer_contact_normal_xy=hammer_contact_normal,
    )


def _discover_repeated_vec2_addr(
    pid: int,
    *,
    roots: dict[str, int],
    samples: list[tuple[float, float]],
    window: int,
    eps: float,
) -> int | None:
    from collections import Counter

    from .live_position import choose_candidate, discover_paths, resolve_candidate_addr

    hits = Counter()
    with MemReader(pid) as reader:
        for x, y in samples:
            hits.update(
                discover_paths(
                    reader,
                    roots=roots,
                    target_x=x,
                    target_y=y,
                    window=window,
                    eps=eps,
                )
            )
        if not hits:
            return None
        candidate, _ = choose_candidate(hits)
        return resolve_candidate_addr(reader, roots, candidate)


def _scan_float_matches(reader: MemReader, roots: dict[str, int], target: float, window: int, eps: float) -> dict[int, tuple[int, int]]:
    matches: dict[int, tuple[int, int]] = {}

    def scan(root_addr: int, indirection_rank: int) -> None:
        region = reader.region_for(root_addr)
        if not region or "r" not in region.perms:
            return
        start = max(region.start, root_addr - window)
        end = min(region.end, root_addr + window)
        data = reader.read(start, end - start)
        for idx in range(0, len(data) - 4, 4):
            value = struct.unpack_from("<f", data, idx)[0]
            if abs(_wrap_angle_delta(value - target)) > eps:
                continue
            addr = start + idx
            distance = abs(addr - root_addr)
            previous = matches.get(addr)
            if previous is None or (indirection_rank, distance) < previous:
                matches[addr] = (indirection_rank, distance)

    for root in roots.values():
        if not root:
            continue
        scan(root, 0)
        region = reader.region_for(root)
        if not region or "r" not in region.perms:
            continue
        head = reader.read(root, min(0x200, region.end - root))
        for ptr_offset in range(0, len(head) - 8, 8):
            ptr = struct.unpack_from("<Q", head, ptr_offset)[0]
            ptr_region = reader.region_for(ptr)
            if not ptr_region or "r" not in ptr_region.perms:
                continue
            scan(ptr, 1)

    return matches


def _discover_repeated_angle_addr(
    pid: int,
    *,
    roots: dict[str, int],
    samples: list[float],
    window: int,
    eps: float,
) -> int | None:
    from collections import Counter

    with MemReader(pid) as reader:
        hits = Counter()
        rankings: dict[int, tuple[int, int]] = {}
        for sample in samples:
            for addr, ranking in _scan_float_matches(reader, roots, sample, window, eps).items():
                hits[addr] += 1
                previous = rankings.get(addr)
                if previous is None or ranking < previous:
                    rankings[addr] = ranking
        if not hits:
            return None
        best_addr, _ = sorted(
            hits.items(),
            key=lambda item: (-item[1], rankings[item[0]][0], rankings[item[0]][1], item[0]),
        )[0]
        return best_addr


def _read_contact_state_from_layout(
    reader: MemReader,
    layout: ResolvedLiveLayout,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    if layout.hammer_contact_flags_addr is None or layout.hammer_contact_normal_addr is None:
        return None, None
    try:
        slide = bool(reader.read_u8(layout.hammer_contact_flags_addr))
        contacts_obj = reader.read_ptr(layout.hammer_contact_normal_addr)
        if not contacts_obj:
            return _decode_hammer_contact_state(slide=slide, contact_count=0, first_contact=None)
        contact_count = int(reader.read_ptr(contacts_obj + 0x18))
        first_contact = reader.read(contacts_obj + 0x20, 16) if contact_count > 0 else None
        return _decode_hammer_contact_state(slide=slide, contact_count=contact_count, first_contact=first_contact)
    except OSError:
        return None, None


def _read_progress_height(reader: MemReader, layout: ResolvedLiveLayout) -> float | None:
    progress_vec = read_optional_vec2(reader, layout.progress_addr)
    if progress_vec is not None:
        return progress_vec[1]
    body_position = read_optional_vec2(reader, layout.body_position_addr)
    if body_position is not None:
        return body_position[1]
    return None


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def resolve_live_layout(
    pid: int,
    *,
    calibration_samples: int,
    calibration_interval: float,
    window: int,
    eps: float,
    startup_timeout: float | None = None,
    resolve_optional_fields: bool = False,
    fast_cursor_addr: int | None = None,
) -> ResolvedLiveLayout:
    from .live_position import freeze_fast_cursor_lane

    deadline = None if startup_timeout is None else time.monotonic() + max(0.0, startup_timeout)
    process, runtime = _attach_runtime(pid)
    try:
        refs = _resolve_playercontrol_refs(runtime, process)

        body_samples: list[tuple[float, float]] = []
        body_angle_samples: list[float] = []
        hammer_anchor_samples: list[tuple[float, float]] = []
        hammer_tip_samples: list[tuple[float, float]] = []
        for sample_index in range(1):
            if sample_index > 0:
                time.sleep(calibration_interval)
            snapshot = _sample_observation_snapshot(runtime, process, refs)
            body_samples.append(snapshot.body_position_xy)
            body_angle_samples.append(snapshot.body_angle)
            hammer_anchor_samples.append(snapshot.hammer_anchor_xy)
            hammer_tip_samples.append(snapshot.hammer_tip_xy)
    finally:
        process.close()

    if fast_cursor_addr is None:
        fast_lane = freeze_fast_cursor_lane(
            pid=pid,
            calibration_samples=calibration_samples,
            calibration_interval=calibration_interval,
            window=window,
            eps=eps,
        )
        fast_cursor_addr = fast_lane.current_addr

    body_roots = {
        "body_native": refs.body_transform_native,
        "body_managed": refs.body_transform_obj,
    }
    hammer_anchor_roots = {
        "hammer_anchor_native": refs.hammer_anchor_transform_native,
        "hammer_anchor_managed": refs.hammer_anchor_transform_obj,
    }
    hammer_tip_roots = {
        "hammer_tip_native": refs.hammer_tip_transform_native,
        "hammer_tip_managed": refs.hammer_tip_transform_obj,
    }

    body_position_addr = None if _deadline_expired(deadline) else _discover_repeated_vec2_addr(
        pid,
        roots=body_roots,
        samples=body_samples,
        window=window,
        eps=eps,
    )
    body_angle_addr = None
    hammer_anchor_addr = None
    hammer_tip_addr = None
    if resolve_optional_fields and not _deadline_expired(deadline):
        body_angle_addr = _discover_repeated_angle_addr(
            pid,
            roots=body_roots,
            samples=body_angle_samples,
            window=window,
            eps=max(eps * 4.0, 0.01),
        )
    if resolve_optional_fields and not _deadline_expired(deadline):
        hammer_anchor_addr = _discover_repeated_vec2_addr(
            pid,
            roots=hammer_anchor_roots,
            samples=hammer_anchor_samples,
            window=window,
            eps=eps,
        )
    if resolve_optional_fields and not _deadline_expired(deadline):
        hammer_tip_addr = _discover_repeated_vec2_addr(
            pid,
            roots=hammer_tip_roots,
            samples=hammer_tip_samples,
            window=window,
            eps=eps,
        )

    hammer_contact_flags_addr = refs.hammer_collisions_obj + 0x4C if refs.hammer_collisions_obj else None
    hammer_contact_normal_addr = refs.hammer_collisions_obj + 0x268 if refs.hammer_collisions_obj else None

    valid_mask = {
        "cursor_position_xy": True,
        "body_position_xy": body_position_addr is not None,
        "body_angle": body_angle_addr is not None,
        "hammer_anchor_xy": hammer_anchor_addr is not None,
        "hammer_tip_xy": hammer_tip_addr is not None,
        "hammer_contact_flags": hammer_contact_flags_addr is not None and hammer_contact_normal_addr is not None,
        "hammer_contact_normal_xy": hammer_contact_flags_addr is not None and hammer_contact_normal_addr is not None,
        "progress_features": body_position_addr is not None,
    }

    return ResolvedLiveLayout(
        pid=pid,
        fast_cursor_addr=fast_cursor_addr,
        body_position_addr=body_position_addr,
        body_angle_addr=body_angle_addr,
        hammer_anchor_addr=hammer_anchor_addr,
        hammer_tip_addr=hammer_tip_addr,
        hammer_contact_flags_addr=hammer_contact_flags_addr,
        hammer_contact_normal_addr=hammer_contact_normal_addr,
        progress_addr=body_position_addr,
        valid_mask=valid_mask,
        discovered_at=time.time(),
    )


def validate_live_layout(pid: int, layout: ResolvedLiveLayout) -> ValidationReport:
    process, runtime = _attach_runtime(pid)
    try:
        refs = _resolve_playercontrol_refs(runtime, process)
        snapshot = _sample_observation_snapshot(runtime, process, refs)
        cursor_truth = runtime.get_position_via_icall(refs.fake_cursor_rb_obj)
    finally:
        process.close()

    mismatches: dict[str, float] = {}
    notes: list[str] = []

    with MemReader(pid) as reader:
        cursor_raw = read_optional_vec2(reader, layout.fast_cursor_addr)
        if cursor_raw is None:
            notes.append("cursor_position_xy unreadable from fast_cursor_addr")
            mismatches["cursor_position_xy"] = float("inf")
        else:
            mismatches["cursor_position_xy"] = max(
                abs(cursor_raw[0] - cursor_truth[0]),
                abs(cursor_raw[1] - cursor_truth[1]),
            )

        if layout.valid_mask.get("body_position_xy"):
            body_raw = read_optional_vec2(reader, layout.body_position_addr)
            if body_raw is None:
                notes.append("body_position_xy marked valid but unreadable")
                mismatches["body_position_xy"] = float("inf")
            else:
                mismatches["body_position_xy"] = max(
                    abs(body_raw[0] - snapshot.body_position_xy[0]),
                    abs(body_raw[1] - snapshot.body_position_xy[1]),
                )
        else:
            notes.append("body_position_xy unresolved")

        if layout.valid_mask.get("body_angle"):
            body_angle_raw = read_optional_f32(reader, layout.body_angle_addr)
            if body_angle_raw is None:
                notes.append("body_angle marked valid but unreadable")
                mismatches["body_angle"] = float("inf")
            else:
                mismatches["body_angle"] = abs(_wrap_angle_delta(body_angle_raw - snapshot.body_angle))
        else:
            notes.append("body_angle unresolved")

        if layout.valid_mask.get("hammer_anchor_xy"):
            hammer_anchor_raw = read_optional_vec2(reader, layout.hammer_anchor_addr)
            if hammer_anchor_raw is None:
                notes.append("hammer_anchor_xy marked valid but unreadable")
                mismatches["hammer_anchor_xy"] = float("inf")
            else:
                mismatches["hammer_anchor_xy"] = max(
                    abs(hammer_anchor_raw[0] - snapshot.hammer_anchor_xy[0]),
                    abs(hammer_anchor_raw[1] - snapshot.hammer_anchor_xy[1]),
                )
        else:
            notes.append("hammer_anchor_xy unresolved")

        if layout.valid_mask.get("hammer_tip_xy"):
            hammer_tip_raw = read_optional_vec2(reader, layout.hammer_tip_addr)
            if hammer_tip_raw is None:
                notes.append("hammer_tip_xy marked valid but unreadable")
                mismatches["hammer_tip_xy"] = float("inf")
            else:
                mismatches["hammer_tip_xy"] = max(
                    abs(hammer_tip_raw[0] - snapshot.hammer_tip_xy[0]),
                    abs(hammer_tip_raw[1] - snapshot.hammer_tip_xy[1]),
                )
        else:
            notes.append("hammer_tip_xy unresolved")

        raw_contact_flags, raw_contact_normal = _read_contact_state_from_layout(reader, layout)
        if layout.valid_mask.get("hammer_contact_flags"):
            if raw_contact_flags is None:
                notes.append("hammer_contact_flags marked valid but unreadable")
                mismatches["hammer_contact_flags"] = float("inf")
            else:
                mismatches["hammer_contact_flags"] = max(
                    abs(raw_contact_flags[0] - snapshot.hammer_contact_flags[0]),
                    abs(raw_contact_flags[1] - snapshot.hammer_contact_flags[1]),
                )
        else:
            notes.append("hammer_contact_flags unresolved")

        if layout.valid_mask.get("hammer_contact_normal_xy"):
            if raw_contact_normal is None:
                notes.append("hammer_contact_normal_xy marked valid but unreadable")
                mismatches["hammer_contact_normal_xy"] = float("inf")
            else:
                mismatches["hammer_contact_normal_xy"] = max(
                    abs(raw_contact_normal[0] - snapshot.hammer_contact_normal_xy[0]),
                    abs(raw_contact_normal[1] - snapshot.hammer_contact_normal_xy[1]),
                )
        else:
            notes.append("hammer_contact_normal_xy unresolved")

        if layout.valid_mask.get("progress_features"):
            progress_height = _read_progress_height(reader, layout)
            if progress_height is None:
                notes.append("progress_features marked valid but unreadable")
                mismatches["progress_features"] = float("inf")
            else:
                mismatches["progress_features"] = abs(progress_height - snapshot.body_position_xy[1])
        else:
            notes.append("progress_features unresolved")

    thresholds = {
        "cursor_position_xy": 0.01,
        "body_position_xy": 0.01,
        "body_angle": 0.05,
        "hammer_anchor_xy": 0.01,
        "hammer_tip_xy": 0.01,
        "hammer_contact_flags": 0.0,
        "hammer_contact_normal_xy": 0.05,
        "progress_features": 0.01,
    }
    ok = all(value <= thresholds.get(name, 0.01) for name, value in mismatches.items())
    return ValidationReport(ok=ok, mismatches=mismatches, notes=notes)


def sample_rigidbody_position(pid: int, rb_obj: int) -> tuple[float, float]:
    process = RemoteProcess(pid)
    try:
        process.attach_all()
        base = gameassembly_base(process.maps)
        fn = process.read_ptr(base + 0x22251B8)
        process.reset_call_frame()
        out_ptr = process.alloc(b"\x00" * 8)
        process.execute(fn, [rb_obj, out_ptr])
        return struct.unpack("<ff", process.read(out_ptr, 8))
    finally:
        process.close()


def find_playercontrol_position(pid: int) -> dict[str, int | float]:
    process, runtime = _attach_runtime(pid)
    try:
        refs = _resolve_playercontrol_refs(runtime, process)
        pos_x, pos_y = runtime.get_position_via_icall(refs.fake_cursor_rb_obj)

        return {
            "player_obj": refs.player_obj,
            "fake_cursor_obj": refs.fake_cursor_obj,
            "fake_cursor_native": refs.fake_cursor_native,
            "fake_cursor_rb_obj": refs.fake_cursor_rb_obj,
            "fake_cursor_rb_native": refs.fake_cursor_rb_native,
            "x": pos_x,
            "y": pos_y,
        }
    finally:
        process.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Use ptrace + IL2CPP exports to query Getting Over It runtime objects.")
    parser.add_argument("--pid", type=int, default=None, help="Target GettingOverIt PID. Defaults to pgrep.")
    parser.add_argument("--watch", action="store_true", help="Continuously sample Rigidbody2D.position after discovery.")
    parser.add_argument("--interval", type=float, default=0.1, help="Seconds between watch samples.")
    parser.add_argument("--samples", type=int, default=0, help="Stop after N watch samples. 0 means run until interrupted.")
    args = parser.parse_args()

    pid = args.pid if args.pid is not None else auto_pid()
    result = find_playercontrol_position(pid)
    print(f"PID: {pid}")
    print(f"PlayerControl:      0x{result['player_obj']:016X}")
    print(f"fakeCursor:         0x{result['fake_cursor_obj']:016X}")
    print(f"fakeCursor native:  0x{result['fake_cursor_native']:016X}")
    print(f"fakeCursorRB:       0x{result['fake_cursor_rb_obj']:016X}")
    print(f"fakeCursorRB native:0x{result['fake_cursor_rb_native']:016X}")
    print(f"Rigidbody2D.position: x={result['x']:.6f} y={result['y']:.6f}")

    if args.watch:
        count = 0
        while args.samples == 0 or count < args.samples:
            x, y = sample_rigidbody_position(pid, int(result["fake_cursor_rb_obj"]))
            print(f"{time.time():.3f} x={x:.6f} y={y:.6f}", flush=True)
            count += 1
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
