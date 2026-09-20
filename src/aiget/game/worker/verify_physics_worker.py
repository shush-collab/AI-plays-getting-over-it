"""Verify that a physics-worker scene preserves collision and controller data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .physics_worker import SCENE_RELATIVE_PATH, load_unitypy, sha256_file

PRESERVED_COMPONENT_TYPES = frozenset(
    {
        "Animator",
        "BoxCollider",
        "BoxCollider2D",
        "Camera",
        "Canvas",
        "CapsuleCollider",
        "CircleCollider2D",
        "Collider",
        "FixedJoint2D",
        "HingeJoint2D",
        "MeshCollider",
        "MonoBehaviour",
        "PolygonCollider2D",
        "Rigidbody2D",
        "SliderJoint2D",
        "SphereCollider",
    }
)


def object_index(scene_path: Path) -> tuple[dict[int, Any], Counter[str]]:
    unitypy = load_unitypy()
    environment = unitypy.load(str(scene_path))
    objects = {
        object_reader.path_id: object_reader for object_reader in environment.file.objects.values()
    }
    return objects, Counter(object_reader.type.name for object_reader in objects.values())


def compare_preserved_components(
    backup_objects: dict[int, Any], worker_objects: dict[int, Any]
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    for path_id, backup_object in backup_objects.items():
        component_type = backup_object.type.name
        if component_type not in PRESERVED_COMPONENT_TYPES:
            continue
        worker_object = worker_objects.get(path_id)
        if worker_object is None or worker_object.type.name != component_type:
            mismatches.append(
                {
                    "path_id": path_id,
                    "component_type": component_type,
                    "reason": "missing_or_type_changed",
                }
            )
            continue
        if hashlib.sha256(backup_object.get_raw_data()).digest() != hashlib.sha256(
            worker_object.get_raw_data()
        ).digest():
            mismatches.append(
                {
                    "path_id": path_id,
                    "component_type": component_type,
                    "reason": "serialized_data_changed",
                }
            )
    return mismatches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, required=True, help="Patched worker game copy.")
    parser.add_argument(
        "--backup-root", type=Path, required=True, help="Untouched reference game backup."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_root = args.game_root.resolve()
    backup_root = args.backup_root.resolve()
    if game_root == backup_root:
        raise SystemExit("game root and backup root must be different directories")
    worker_scene = game_root / SCENE_RELATIVE_PATH
    backup_scene = backup_root / SCENE_RELATIVE_PATH
    if not worker_scene.is_file() or not backup_scene.is_file():
        raise SystemExit(f"missing required scene file {SCENE_RELATIVE_PATH}")

    backup_objects, backup_counts = object_index(backup_scene)
    worker_objects, worker_counts = object_index(worker_scene)
    count_mismatches = {
        component_type: {
            "backup": backup_counts[component_type],
            "worker": worker_counts[component_type],
        }
        for component_type in sorted(PRESERVED_COMPONENT_TYPES)
        if backup_counts[component_type] != worker_counts[component_type]
    }
    component_mismatches = compare_preserved_components(backup_objects, worker_objects)
    report = {
        "backup_scene_sha256": sha256_file(backup_scene),
        "worker_scene_sha256": sha256_file(worker_scene),
        "preserved_component_count_mismatches": count_mismatches,
        "preserved_component_serialized_mismatches": component_mismatches,
        "ok": not count_mismatches and not component_mismatches,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
