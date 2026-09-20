from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aiget.game.worker.physics_worker import (  # noqa: E402
    NULL_POINTER,
    detach_presentation_asset_references,
    disable_presentation_components,
)


class FakeType:
    def __init__(self, name: str):
        self.name = name


class FakeComponent:
    def __init__(self, name: str, enabled: bool = True, **fields):
        self.type = FakeType(name)
        self.tree = {"m_Enabled": enabled, **fields}
        self.saved = False

    def read_typetree(self):
        return dict(self.tree)

    def save_typetree(self, tree):
        self.tree = tree
        self.saved = True


class PhysicsWorkerTests(unittest.TestCase):
    def test_disable_presentation_components_preserves_physics_and_gameplay(self) -> None:
        renderer = FakeComponent("MeshRenderer")
        sound = FakeComponent("AudioSource")
        collider = FakeComponent("PolygonCollider2D")
        rigidbody = FakeComponent("Rigidbody2D")
        script = FakeComponent("MonoBehaviour")
        camera = FakeComponent("Camera")

        disabled = disable_presentation_components(
            [renderer, sound, collider, rigidbody, script, camera]
        )

        self.assertEqual(disabled, {"AudioSource": 1, "MeshRenderer": 1})
        self.assertFalse(renderer.tree["m_Enabled"])
        self.assertFalse(sound.tree["m_Enabled"])
        for component in (collider, rigidbody, script, camera):
            self.assertTrue(component.tree["m_Enabled"])
            self.assertFalse(component.saved)

    def test_detach_presentation_assets_leaves_physics_references_intact(self) -> None:
        mesh_filter = FakeComponent(
            "MeshFilter", m_Mesh={"m_FileID": 2, "m_PathID": 20}
        )
        renderer = FakeComponent(
            "MeshRenderer", m_Materials=[{"m_FileID": 2, "m_PathID": 21}]
        )
        audio = FakeComponent(
            "AudioSource", m_audioClip={"m_FileID": 2, "m_PathID": 22}
        )
        collider = FakeComponent(
            "PolygonCollider2D", m_PhysicsMaterial={"m_FileID": 2, "m_PathID": 23}
        )

        detached = detach_presentation_asset_references(
            [mesh_filter, renderer, audio, collider]
        )

        self.assertEqual(
            detached,
            {
                "AudioSource.m_audioClip": 1,
                "MeshFilter.m_Mesh": 1,
                "MeshRenderer.m_Materials": 1,
            },
        )
        self.assertEqual(mesh_filter.tree["m_Mesh"], NULL_POINTER)
        self.assertEqual(renderer.tree["m_Materials"], [])
        self.assertEqual(audio.tree["m_audioClip"], NULL_POINTER)
        self.assertEqual(collider.tree["m_PhysicsMaterial"], {"m_FileID": 2, "m_PathID": 23})
        self.assertFalse(collider.saved)
