import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.memory_probe import GAME_BINARY, candidate_pids_from_pgrep_output
from aiget.ptrace_il2cpp import (
    ValidationReport,
    choose_int3_trap_offset,
    load_export_offsets,
    resolve_live_layout,
    validate_live_layout,
)
from aiget.live_layout import ResolvedLiveLayout


class _FakeReader:
    def __init__(self, *, vec2=None, f32=None, u8=None, ptr=None, raw=None):
        self.vec2 = vec2 or {}
        self.f32 = f32 or {}
        self.u8 = u8 or {}
        self.ptr = ptr or {}
        self.raw = raw or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read_vec2(self, addr: int):
        return self.vec2[addr]

    def read_f32(self, addr: int):
        return self.f32[addr]

    def read_u8(self, addr: int):
        return self.u8[addr]

    def read_ptr(self, addr: int):
        return self.ptr[addr]

    def read(self, addr: int, size: int):
        return self.raw[addr][:size]


class PtraceIcallTests(unittest.TestCase):
    def tearDown(self) -> None:
        import aiget.ptrace_il2cpp as ptrace_module

        ptrace_module._EXPORT_OFFSETS_CACHE = None

    def test_choose_int3_trap_offset_prefers_nearest_prior_breakpoint_on_tie(self) -> None:
        code = bytes([0x90, 0xCC, 0x90, 0xCC, 0x90])

        trap_offset = choose_int3_trap_offset(code, pivot=2)

        self.assertEqual(trap_offset, 1)

    def test_choose_int3_trap_offset_returns_none_when_missing(self) -> None:
        self.assertIsNone(choose_int3_trap_offset(bytes([0x90, 0x90, 0x90]), pivot=1))

    def test_candidate_pids_from_pgrep_output_filters_wrapper_processes(self) -> None:
        output = "\n".join(
            [
                f"12818 /bin/sh -c {GAME_BINARY}",
                "12819 /home/fln/.steam/debian-installation/ubuntu12_32/reaper SteamLaunch AppId=240720 -- ...",
                f"12898 {GAME_BINARY}",
            ]
        )

        candidates = candidate_pids_from_pgrep_output(output)

        self.assertEqual(candidates, [12898])

    def test_load_export_offsets_caches_first_nm_result(self) -> None:
        nm_stdout = "\n".join(
            [
                "0000000000001000 T il2cpp_assembly_get_image",
                "0000000000001001 T il2cpp_class_from_name",
                "0000000000001002 T il2cpp_class_get_field_from_name",
                "0000000000001003 T il2cpp_class_get_method_from_name",
                "0000000000001004 T il2cpp_class_get_type",
                "0000000000001005 T il2cpp_domain_get",
                "0000000000001006 T il2cpp_domain_get_assemblies",
                "0000000000001007 T il2cpp_field_get_offset",
                "0000000000001008 T il2cpp_field_get_value",
                "0000000000001009 T il2cpp_image_get_name",
                "0000000000001010 T il2cpp_resolve_icall",
                "0000000000001011 T il2cpp_runtime_invoke",
                "0000000000001012 T il2cpp_thread_attach",
                "0000000000001013 T il2cpp_type_get_object",
            ]
        )

        with patch(
            "aiget.ptrace_il2cpp.subprocess.run",
            return_value=SimpleNamespace(stdout=nm_stdout),
        ) as run:
            first = load_export_offsets()
            second = load_export_offsets()

        self.assertEqual(first, second)
        run.assert_called_once()

    def test_validate_live_layout_returns_validation_report_shape(self) -> None:
        layout = ResolvedLiveLayout(
            pid=15081,
            fast_cursor_addr=0x1000,
            body_position_addr=0x2000,
            body_angle_addr=None,
            hammer_anchor_addr=0x3000,
            hammer_tip_addr=0x4000,
            hammer_contact_flags_addr=0x5000,
            hammer_contact_normal_addr=0x6000,
            progress_addr=0x2000,
            valid_mask={
                "cursor_position_xy": True,
                "body_position_xy": True,
                "body_angle": False,
                "hammer_anchor_xy": True,
                "hammer_tip_xy": True,
                "hammer_contact_flags": True,
                "hammer_contact_normal_xy": True,
                "progress_features": True,
            },
            discovered_at=25.0,
        )
        authoritative_snapshot = SimpleNamespace(
            body_position_xy=(1.0, 2.0),
            body_angle=0.25,
            hammer_anchor_xy=(3.0, 4.0),
            hammer_tip_xy=(5.0, 6.0),
            hammer_contact_flags=(1.0, 1.0),
            hammer_contact_normal_xy=(-0.5, 0.75),
        )
        first_contact = bytes.fromhex("0000803f00000040000000bf0000403f")
        fake_reader = _FakeReader(
            vec2={
                0x1000: (9.0, 10.0),
                0x2000: (1.0, 2.0),
                0x3000: (3.0, 4.0),
                0x4000: (5.0, 6.0),
            },
            u8={0x5000: 1},
            ptr={0x6000: 0x7000, 0x7000 + 0x18: 1},
            raw={0x7000 + 0x20: first_contact},
        )

        fake_runtime = SimpleNamespace(get_position_via_icall=lambda rb_obj: (9.0, 10.0))

        with (
            patch("aiget.ptrace_il2cpp._attach_runtime", return_value=(SimpleNamespace(close=lambda: None), fake_runtime)),
            patch("aiget.ptrace_il2cpp._resolve_playercontrol_refs", return_value=SimpleNamespace(fake_cursor_rb_obj=0xDEAD)),
            patch("aiget.ptrace_il2cpp._sample_observation_snapshot", return_value=authoritative_snapshot),
            patch("aiget.ptrace_il2cpp.MemReader", return_value=fake_reader),
        ):
            report = validate_live_layout(15081, layout)

        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(isinstance(report.ok, bool))
        self.assertIn("cursor_position_xy", report.mismatches)
        self.assertIn("body_angle unresolved", report.notes)

    def test_resolve_live_layout_can_fail_fast_with_partial_layout(self) -> None:
        refs = SimpleNamespace(
            body_transform_native=0x1000,
            body_transform_obj=0x2000,
            hammer_anchor_transform_native=0x3000,
            hammer_anchor_transform_obj=0x4000,
            hammer_tip_transform_native=0x5000,
            hammer_tip_transform_obj=0x6000,
            hammer_collisions_obj=0x7000,
        )
        snapshot = SimpleNamespace(
            body_position_xy=(1.0, 2.0),
            body_angle=0.25,
            hammer_anchor_xy=(3.0, 4.0),
            hammer_tip_xy=(5.0, 6.0),
        )
        fake_process = SimpleNamespace(close=lambda: None)

        with (
            patch("aiget.ptrace_il2cpp._attach_runtime", return_value=(fake_process, object())),
            patch("aiget.ptrace_il2cpp._resolve_playercontrol_refs", return_value=refs),
            patch("aiget.ptrace_il2cpp._sample_observation_snapshot", return_value=snapshot),
            patch("aiget.ptrace_il2cpp._discover_repeated_vec2_addr") as discover_vec2,
            patch("aiget.live_position.freeze_fast_cursor_lane") as freeze_fast_lane,
        ):
            layout = resolve_live_layout(
                15081,
                calibration_samples=1,
                calibration_interval=0.7,
                window=0x200,
                eps=0.0015,
                startup_timeout=0.0,
                resolve_optional_fields=False,
                fast_cursor_addr=0xABC,
            )

        discover_vec2.assert_not_called()
        freeze_fast_lane.assert_not_called()
        self.assertEqual(layout.fast_cursor_addr, 0xABC)
        self.assertIsNone(layout.body_position_addr)
        self.assertFalse(layout.valid_mask["body_position_xy"])
        self.assertFalse(layout.valid_mask["hammer_anchor_xy"])
        self.assertTrue(layout.valid_mask["hammer_contact_flags"])


if __name__ == "__main__":
    unittest.main()
