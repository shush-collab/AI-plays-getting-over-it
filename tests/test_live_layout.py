import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.live_layout import ResolvedLiveLayout, default_live_layout_cache_path, load_live_layout, save_live_layout


class LiveLayoutTests(unittest.TestCase):
    def test_default_live_layout_cache_path_includes_pid(self) -> None:
        path = default_live_layout_cache_path(9876)

        self.assertIn(".cache", path)
        self.assertIn("aiget", path)
        self.assertIn("9876", path)

    def test_layout_save_and_load_round_trip(self) -> None:
        layout = ResolvedLiveLayout(
            pid=1234,
            fast_cursor_addr=0x1111,
            body_position_addr=0x2222,
            body_angle_addr=None,
            hammer_anchor_addr=0x3333,
            hammer_tip_addr=0x4444,
            hammer_contact_flags_addr=0x5555,
            hammer_contact_normal_addr=0x6666,
            progress_addr=0x2222,
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
            discovered_at=99.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "layout.json"
            save_live_layout(str(path), layout)
            loaded = load_live_layout(str(path))

        self.assertEqual(loaded, layout)

    def test_unresolved_fields_can_be_persisted_with_valid_mask_false(self) -> None:
        layout = ResolvedLiveLayout(
            pid=4321,
            fast_cursor_addr=0xAAAA,
            body_position_addr=None,
            body_angle_addr=None,
            hammer_anchor_addr=0xBBBB,
            hammer_tip_addr=None,
            hammer_contact_flags_addr=None,
            hammer_contact_normal_addr=None,
            progress_addr=None,
            valid_mask={
                "cursor_position_xy": True,
                "body_position_xy": False,
                "body_angle": False,
                "hammer_anchor_xy": True,
                "hammer_tip_xy": False,
                "hammer_contact_flags": False,
                "hammer_contact_normal_xy": False,
                "progress_features": False,
            },
            discovered_at=12.25,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "layout.json"
            save_live_layout(str(path), layout)
            loaded = load_live_layout(str(path))

        self.assertIsNone(loaded.body_position_addr)
        self.assertIsNone(loaded.body_angle_addr)
        self.assertFalse(loaded.valid_mask["body_position_xy"])
        self.assertFalse(loaded.valid_mask["progress_features"])


if __name__ == "__main__":
    unittest.main()
