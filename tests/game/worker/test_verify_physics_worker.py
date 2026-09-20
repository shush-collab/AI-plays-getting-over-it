from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.game.worker.verify_physics_worker import (  # noqa: E402
    PRESERVED_COMPONENT_TYPES,
    compare_preserved_components,
)


class FakeType:
    def __init__(self, name: str):
        self.name = name


class FakeObject:
    def __init__(self, name: str, raw_data: bytes):
        self.type = FakeType(name)
        self._raw_data = raw_data

    def get_raw_data(self) -> bytes:
        return self._raw_data


class VerifyPhysicsWorkerTests(unittest.TestCase):
    def test_preserved_components_include_2d_physics_and_gameplay(self) -> None:
        self.assertTrue(
            {
                "PolygonCollider2D",
                "Rigidbody2D",
                "HingeJoint2D",
                "SliderJoint2D",
                "MonoBehaviour",
            }.issubset(PRESERVED_COMPONENT_TYPES)
        )

    def test_compare_preserved_components_reports_only_relevant_changes(self) -> None:
        backup = {
            1: FakeObject("PolygonCollider2D", b"collision"),
            2: FakeObject("MonoBehaviour", b"controller"),
            3: FakeObject("MeshRenderer", b"presentation"),
        }
        worker = {
            1: FakeObject("PolygonCollider2D", b"collision"),
            2: FakeObject("MonoBehaviour", b"changed"),
            3: FakeObject("MeshRenderer", b"removed-material"),
        }

        self.assertEqual(
            compare_preserved_components(backup, worker),
            [
                {
                    "path_id": 2,
                    "component_type": "MonoBehaviour",
                    "reason": "serialized_data_changed",
                }
            ],
        )
