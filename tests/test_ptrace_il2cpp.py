import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.memory_probe import GAME_BINARY, candidate_pids_from_pgrep_output
from aiget.ptrace_il2cpp import choose_int3_trap_offset


class PtraceIcallTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
