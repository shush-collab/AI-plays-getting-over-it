#!/usr/bin/env python3
import argparse
import ctypes
import ctypes.util
import os
import signal
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

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
    return offsets


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
        self.process.reset_call_frame()
        name_ptr = self.process.alloc_cstring(name)
        regs = self.process.execute(self.exports["il2cpp_resolve_icall"], [name_ptr])
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
        object_class = runtime.class_from_name(unity_core, "UnityEngine", "Object")
        if not player_class or not object_class:
            raise RemoteError("PlayerControl or UnityEngine.Object class not found")

        player_type = runtime.call("il2cpp_class_get_type", [player_class])
        player_type_obj = runtime.call("il2cpp_type_get_object", [player_type])
        find_object = runtime.method_from_name(object_class, "FindObjectOfType", 1)
        if not find_object:
            raise RemoteError("UnityEngine.Object.FindObjectOfType(Type) method not found")

        player_obj = runtime.invoke(find_object, 0, [player_type_obj])
        if not player_obj:
            raise RemoteError("FindObjectOfType(PlayerControl) returned null")

        fake_cursor_rb_field = runtime.field_from_name(player_class, "fakeCursorRB")
        fake_cursor_field = runtime.field_from_name(player_class, "fakeCursor")
        if not fake_cursor_rb_field or not fake_cursor_field:
            raise RemoteError("fakeCursorRB or fakeCursor field lookup failed")

        fake_cursor_rb_obj = runtime.get_field_value_ptr(player_obj, fake_cursor_rb_field)
        fake_cursor_obj = runtime.get_field_value_ptr(player_obj, fake_cursor_field)
        fake_cursor_rb_native = process.read_ptr(fake_cursor_rb_obj + 0x10) if fake_cursor_rb_obj else 0
        fake_cursor_native = process.read_ptr(fake_cursor_obj + 0x10) if fake_cursor_obj else 0
        pos_x, pos_y = runtime.get_position_via_icall(fake_cursor_rb_obj)

        return {
            "player_obj": player_obj,
            "fake_cursor_obj": fake_cursor_obj,
            "fake_cursor_native": fake_cursor_native,
            "fake_cursor_rb_obj": fake_cursor_rb_obj,
            "fake_cursor_rb_native": fake_cursor_rb_native,
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
