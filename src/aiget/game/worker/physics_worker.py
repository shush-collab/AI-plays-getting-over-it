"""Build a presentation-free Unity scene copy for Getting Over It worker experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCENE_RELATIVE_PATH = Path("GettingOverIt_Data/level1")
PRESENTATION_COMPONENT_TYPES = frozenset(
    {
        "AudioListener",
        "AudioSource",
        "BillboardRenderer",
        "Light",
        "MeshRenderer",
        "ParticleSystemRenderer",
        "ReflectionProbe",
        "SkinnedMeshRenderer",
        "SpriteRenderer",
    }
)

# These serialized pointers are used only to render, emit particles, or play audio.  They
# are deliberately kept separate from collider, Rigidbody2D, joint, and MonoBehaviour data.
PRESENTATION_REFERENCE_ARRAY_FIELDS: dict[str, frozenset[str]] = {
    "MeshRenderer": frozenset({"m_Materials"}),
    "ParticleSystemRenderer": frozenset({"m_Materials"}),
    "SkinnedMeshRenderer": frozenset({"m_Materials"}),
    "SpriteRenderer": frozenset({"m_Materials"}),
}
PRESENTATION_REFERENCE_FIELDS: dict[str, frozenset[str]] = {
    "AudioSource": frozenset({"m_audioClip"}),
    "BillboardRenderer": frozenset({"m_Billboard"}),
    "MeshFilter": frozenset({"m_Mesh"}),
    "ParticleSystemRenderer": frozenset({"m_Mesh", "m_Mesh1", "m_Mesh2", "m_Mesh3"}),
    "ReflectionProbe": frozenset({"m_BakedTexture", "m_CustomBakedTexture"}),
    "SkinnedMeshRenderer": frozenset({"m_Mesh"}),
    "SpriteRenderer": frozenset({"m_Sprite"}),
}
NULL_POINTER = {"m_FileID": 0, "m_PathID": 0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_type_name(component: Any) -> str:
    return str(component.type.name)


def disable_presentation_components(components: Iterable[Any]) -> Counter[str]:
    """Disable only components whose enabled state is presentation-only.

    Physics components, gameplay scripts, cameras, canvases, and animators are deliberately
    excluded. Some gameplay scripts retain renderer references, so deleting those components
    would be unsafe before live equivalence checks.
    """
    disabled: Counter[str] = Counter()
    for component in components:
        component_type = component_type_name(component)
        if component_type not in PRESENTATION_COMPONENT_TYPES:
            continue
        tree = component.read_typetree()
        if not tree.get("m_Enabled", False):
            continue
        tree["m_Enabled"] = False
        component.save_typetree(tree)
        disabled[component_type] += 1
    return disabled


def detach_presentation_asset_references(components: Iterable[Any]) -> Counter[str]:
    """Clear only serialized visual/audio asset pointers from a scene.

    A disabled Renderer still keeps its material and mesh reachable. Clearing those links
    lets ``Resources.UnloadUnusedAssets`` discard presentation data in a headless worker.
    This function never visits colliders, Rigidbody2D, joints, cameras, animators, or scripts.
    """
    detached: Counter[str] = Counter()
    for component in components:
        component_type = component_type_name(component)
        array_fields = PRESENTATION_REFERENCE_ARRAY_FIELDS.get(component_type, frozenset())
        pointer_fields = PRESENTATION_REFERENCE_FIELDS.get(component_type, frozenset())
        if not array_fields and not pointer_fields:
            continue

        tree = component.read_typetree()
        changed = False
        for field in array_fields:
            if tree.get(field):
                tree[field] = []
                detached[f"{component_type}.{field}"] += 1
                changed = True
        for field in pointer_fields:
            if tree.get(field) and tree[field] != NULL_POINTER:
                tree[field] = dict(NULL_POINTER)
                detached[f"{component_type}.{field}"] += 1
                changed = True
        if changed:
            component.save_typetree(tree)
    return detached


def inspect_scene(components: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(component_type_name(component) for component in components).items()))


def load_unitypy() -> Any:
    try:
        import UnityPy
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "UnityPy is required. Install the game-worker extra before using this command."
        ) from exc
    return UnityPy


def resolve_scene_paths(*, game_root: Path, backup_root: Path) -> tuple[Path, Path]:
    game_root = game_root.resolve()
    backup_root = backup_root.resolve()
    if game_root == backup_root:
        raise RuntimeError("game root and backup root must be different directories")
    scene_path = game_root / SCENE_RELATIVE_PATH
    backup_scene_path = backup_root / SCENE_RELATIVE_PATH
    if not scene_path.is_file() or not backup_scene_path.is_file():
        raise RuntimeError(f"missing required scene file {SCENE_RELATIVE_PATH}")
    return scene_path, backup_scene_path


def rebuild_scene_from_backup(*, scene_path: Path, backup_scene_path: Path) -> None:
    temporary_path = scene_path.with_suffix(scene_path.suffix + ".physics-worker.restore.tmp")
    shutil.copy2(backup_scene_path, temporary_path)
    os.replace(temporary_path, scene_path)


def write_scene(environment: Any, scene_path: Path) -> None:
    temporary_path = scene_path.with_suffix(scene_path.suffix + ".physics-worker.tmp")
    temporary_path.write_bytes(environment.file.save())
    os.replace(temporary_path, scene_path)


def build_manifest(
    *,
    scene_path: Path,
    before: dict[str, int],
    disabled: Counter[str],
    detached: Counter[str],
    applied: bool,
    rebuilt_from_backup: bool,
) -> dict[str, object]:
    return {
        "scene": str(scene_path),
        "scene_sha256": sha256_file(scene_path),
        "applied": applied,
        "rebuilt_from_backup": rebuilt_from_backup,
        "component_counts_before": before,
        "disabled_components": dict(sorted(disabled.items())),
        "detached_presentation_references": dict(sorted(detached.items())),
        "preserved": [
            "all Collider2D and Collider components",
            "all Rigidbody2D components",
            "all Joint2D components",
            "all MonoBehaviour scripts",
            "all Camera, Canvas, and Animator components",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, required=True, help="Editable worker game copy.")
    parser.add_argument("--backup-root", type=Path, help="Verified untouched game backup.")
    parser.add_argument(
        "--apply", action="store_true", help="Disable the safe presentation components."
    )
    parser.add_argument(
        "--detach-presentation-assets",
        action="store_true",
        help="Clear visual/audio scene references so Unity can unload their assets.",
    )
    parser.add_argument(
        "--rebuild-from-backup",
        action="store_true",
        help="Restore level1 from the verified backup before applying the requested worker patch.",
    )
    parser.add_argument("--manifest", type=Path, help="Optional path for the JSON result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_root = args.game_root.resolve()
    scene_path = game_root / SCENE_RELATIVE_PATH
    if not scene_path.is_file():
        raise SystemExit(f"scene not found: {scene_path}")
    if (args.apply or args.rebuild_from_backup) and args.backup_root is None:
        raise SystemExit("--apply and --rebuild-from-backup require --backup-root")
    if args.rebuild_from_backup and not args.apply:
        raise SystemExit("--rebuild-from-backup requires --apply")

    disabled: Counter[str] = Counter()
    detached: Counter[str] = Counter()
    if args.apply:
        scene_path, backup_scene_path = resolve_scene_paths(
            game_root=game_root, backup_root=args.backup_root.resolve()
        )
        if args.rebuild_from_backup:
            rebuild_scene_from_backup(scene_path=scene_path, backup_scene_path=backup_scene_path)
        elif sha256_file(scene_path) != sha256_file(backup_scene_path):
            raise SystemExit("worker scene no longer matches its backup; use --rebuild-from-backup")

    unitypy = load_unitypy()
    environment = unitypy.load(str(scene_path))
    components = list(environment.file.objects.values())
    before = inspect_scene(components)
    if args.apply:
        disabled = disable_presentation_components(components)
        if args.detach_presentation_assets:
            detached = detach_presentation_asset_references(components)
        write_scene(environment, scene_path)

    manifest = build_manifest(
        scene_path=scene_path,
        before=before,
        disabled=disabled,
        detached=detached,
        applied=args.apply,
        rebuilt_from_backup=args.rebuild_from_backup,
    )
    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    if args.manifest is not None:
        args.manifest.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
