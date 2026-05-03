#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ptrace_il2cpp import ValidationReport


@dataclass(frozen=True)
class ResolvedLiveLayout:
    pid: int
    fast_cursor_addr: int
    body_position_addr: int | None
    body_angle_addr: int | None
    hammer_anchor_addr: int | None
    hammer_tip_addr: int | None
    hammer_contact_flags_addr: int | None
    hammer_contact_normal_addr: int | None
    progress_addr: int | None
    valid_mask: dict[str, bool]
    discovered_at: float


def default_live_layout_cache_path(pid: int) -> str:
    return str(Path.home() / ".cache" / "aiget" / f"live-layout-{pid}.json")


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
    from .ptrace_il2cpp import resolve_live_layout as _resolve_live_layout

    return _resolve_live_layout(
        pid,
        calibration_samples=calibration_samples,
        calibration_interval=calibration_interval,
        window=window,
        eps=eps,
        startup_timeout=startup_timeout,
        resolve_optional_fields=resolve_optional_fields,
        fast_cursor_addr=fast_cursor_addr,
    )


def save_live_layout(path: str, layout: ResolvedLiveLayout) -> None:
    payload = {
        "pid": layout.pid,
        "fast_cursor_addr": layout.fast_cursor_addr,
        "body_position_addr": layout.body_position_addr,
        "body_angle_addr": layout.body_angle_addr,
        "hammer_anchor_addr": layout.hammer_anchor_addr,
        "hammer_tip_addr": layout.hammer_tip_addr,
        "hammer_contact_flags_addr": layout.hammer_contact_flags_addr,
        "hammer_contact_normal_addr": layout.hammer_contact_normal_addr,
        "progress_addr": layout.progress_addr,
        "valid_mask": layout.valid_mask,
        "discovered_at": layout.discovered_at,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_live_layout(path: str) -> ResolvedLiveLayout:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ResolvedLiveLayout(
        pid=int(payload["pid"]),
        fast_cursor_addr=int(payload["fast_cursor_addr"]),
        body_position_addr=_optional_int(payload.get("body_position_addr")),
        body_angle_addr=_optional_int(payload.get("body_angle_addr")),
        hammer_anchor_addr=_optional_int(payload.get("hammer_anchor_addr")),
        hammer_tip_addr=_optional_int(payload.get("hammer_tip_addr")),
        hammer_contact_flags_addr=_optional_int(payload.get("hammer_contact_flags_addr")),
        hammer_contact_normal_addr=_optional_int(payload.get("hammer_contact_normal_addr")),
        progress_addr=_optional_int(payload.get("progress_addr")),
        valid_mask={str(key): bool(value) for key, value in dict(payload.get("valid_mask", {})).items()},
        discovered_at=float(payload["discovered_at"]),
    )


def validate_live_layout(pid: int, layout: ResolvedLiveLayout) -> "ValidationReport":
    from .ptrace_il2cpp import validate_live_layout as _validate_live_layout

    return _validate_live_layout(pid, layout)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
