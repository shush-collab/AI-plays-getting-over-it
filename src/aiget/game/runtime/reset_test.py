"""Relaunch and validate an owned Getting Over It installation without Gymnasium."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...shared.png_utils import write_gray_png
from ..probing.live_layout import ResolvedLiveLayout, resolve_live_layout
from ..probing.memory_probe import auto_pid
from .cli_utils import (
    add_capture_region_args,
    capture_region_from_args,
    parse_args_allowing_launch_flags,
)
from .frame_capture import CaptureRegion, FrameCapture
from .reset_startup import StartupAutomation, StartupEvent
from .window_control import WindowControl

RESET_MODE = "relaunch_save_restore"


@dataclass(frozen=True)
class ResetConfig:
    launch_command: tuple[str, ...]
    clean_save_path: Path
    active_save_path: Path
    capture_region: CaptureRegion
    game_ready_timeout: float
    image_std_threshold: float
    image_mean_min: float
    image_mean_max: float
    startup_input_backend: str
    startup_reference_dir: Path | None
    startup_title_click: tuple[int, int]
    startup_confirm_click: tuple[int, int]
    startup_key: str | None
    startup_attempts: int
    startup_probe_timeout: float
    window_title: str
    window_left: int | None
    window_top: int | None
    window_width: int | None
    window_height: int | None
    kill_existing: bool
    calibration_samples: int
    calibration_interval: float
    layout_window: int
    layout_eps: float


@dataclass(frozen=True)
class ResetResult:
    pid: int
    frame: np.ndarray
    trace: dict[str, object]


class ResetFailure(RuntimeError):
    """A reset failure that retains the structured trace for diagnostics."""

    def __init__(self, message: str, trace: dict[str, object]):
        super().__init__(message)
        self.trace = trace


class RelaunchResetTester:
    """Restore a checkpoint, relaunch the game, and wait until its player rig resolves."""

    def __init__(self, config: ResetConfig):
        self.config = config
        self.last_frame: np.ndarray | None = None
        self.startup_frames: list[tuple[str, np.ndarray]] = []
        self.last_trace: dict[str, object] = {}
        self._pid: int | None = None

    def run_once(self) -> ResetResult:
        started = time.perf_counter()
        trace: dict[str, object] = {"reset_mode": RESET_MODE}
        capture: FrameCapture | None = None
        self.last_frame = None
        self.startup_frames = []
        self.last_trace = trace
        try:
            trace["old_pid"] = _find_existing_pid()
            if self.config.kill_existing:
                kill_existing_game()
            restore_save(self.config.clean_save_path, self.config.active_save_path)
            subprocess.Popen(self.config.launch_command)
            self._pid = wait_for_game_process(self.config.game_ready_timeout, trace, started)
            control = wait_for_window(self.config, trace, started)
            capture = FrameCapture(
                output_shape=(
                    self.config.capture_region.height,
                    self.config.capture_region.width,
                    1,
                ),
                region=self.config.capture_region,
                allow_blank=False,
                prefer_xwd=False,
            )
            layout = self._drive_startup(capture, control, trace, started)
            frame = self._read_frame(capture)
            validate_playable_frame(frame, self.config)
            trace["image_ready_ms"] = elapsed_ms(started)
            trace["image_mean"] = float(frame.mean())
            trace["image_std"] = float(frame.std())
            trace["pid"] = self._pid
            trace["fast_cursor_addr"] = f"0x{layout.fast_cursor_addr:016X}"
            self.last_trace = trace
            return ResetResult(pid=self._pid, frame=frame, trace=dict(trace))
        except Exception as exc:
            trace.setdefault("failure_stage", failure_stage(exc))
            trace["reason"] = str(exc)
            self.last_trace = trace
            raise ResetFailure(str(exc), dict(trace)) from exc
        finally:
            if capture is not None:
                capture.close()

    def cleanup(self) -> None:
        if self.config.kill_existing:
            kill_existing_game()

    def _drive_startup(
        self,
        capture: FrameCapture,
        control: WindowControl,
        trace: dict[str, object],
        started: float,
    ) -> ResolvedLiveLayout:
        automation = StartupAutomation(
            frame_getter=lambda: self._read_frame(capture),
            full_frame_getter=lambda: self._read_frame(capture),
            window_control=control,
            resolve_layout=lambda: self._resolve_layout(trace, started),
            timeout=self.config.game_ready_timeout,
            reference_dir=self.config.startup_reference_dir,
            action_interval=2.0,
            max_actions=self.config.startup_attempts,
            title_click=self.config.startup_title_click,
            confirm_click=self.config.startup_confirm_click,
            startup_key=self.config.startup_key,
        )
        result = automation.drive_until_playable()
        self._save_startup_frames(result.events)
        trace["startup"] = result.as_trace()
        trace["startup_state"] = result.state.value
        trace["startup_actions_sent"] = result.attempts
        trace["startup_action_sent"] = result.attempts > 0
        try:
            trace["startup_input_backend_resolved"] = control.active_input_backend
        except RuntimeError:
            pass
        if not result.success or not isinstance(result.layout, ResolvedLiveLayout):
            trace["failure_stage"] = result.reason or "PLAYERCONTROL_TIMEOUT"
            raise RuntimeError(trace["failure_stage"])
        trace["playercontrol_ready_ms"] = elapsed_ms(started)
        return result.layout

    def _resolve_layout(
        self,
        trace: dict[str, object],
        started: float,
    ) -> ResolvedLiveLayout:
        pid = auto_pid()
        if pid != self._pid:
            self._pid = pid
            trace["pid_replaced"] = pid
        if not game_modules_ready(pid):
            raise RuntimeError("MODULES_TIMEOUT")
        trace["startup_probe_attempts"] = int(trace.get("startup_probe_attempts", 0)) + 1
        return resolve_live_layout(
            pid,
            calibration_samples=self.config.calibration_samples,
            calibration_interval=self.config.calibration_interval,
            window=self.config.layout_window,
            eps=self.config.layout_eps,
            startup_timeout=self.config.startup_probe_timeout,
        )

    def _read_frame(self, capture: FrameCapture) -> np.ndarray:
        frame = capture.read_full_gray()
        self.last_frame = frame
        return frame

    def _save_startup_frames(self, events: list[StartupEvent]) -> None:
        for event in events:
            if event.action == "resolved":
                self.startup_frames.append(("playable_full", event.frame.copy()))
                continue
            self.startup_frames.append((f"before_click_{event.index}_full", event.frame.copy()))
            if event.after_frame is not None:
                self.startup_frames.append(
                    (f"after_click_{event.index}_full", event.after_frame.copy())
                )


def restore_save(clean_path: Path, active_path: Path) -> None:
    """Replace the active save with a known clean directory or file."""
    clean = clean_path.expanduser()
    active = active_path.expanduser()
    if not clean.exists():
        raise RuntimeError(f"clean save path does not exist: {clean}")
    active.parent.mkdir(parents=True, exist_ok=True)
    if clean.is_dir():
        if active.exists() and not active.is_dir():
            active.unlink()
        if active.exists():
            shutil.rmtree(active)
        shutil.copytree(clean, active)
        return
    if active.exists() and active.is_dir():
        shutil.rmtree(active)
    shutil.copy2(clean, active)


def wait_for_game_process(
    timeout: float,
    trace: dict[str, object],
    started: float,
) -> int:
    deadline = time.monotonic() + timeout
    process_seen = False
    while time.monotonic() < deadline:
        try:
            pid = auto_pid()
        except RuntimeError:
            time.sleep(0.5)
            continue
        if not process_seen:
            trace["process_ready_ms"] = elapsed_ms(started)
            process_seen = True
        if game_modules_ready(pid):
            trace["modules_ready_ms"] = elapsed_ms(started)
            return pid
        time.sleep(0.5)
    raise RuntimeError("MODULES_TIMEOUT")


def game_modules_ready(pid: int, proc_root: Path = Path("/proc")) -> bool:
    try:
        maps = (proc_root / str(pid) / "maps").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "GameAssembly.so" in maps and "UnityPlayer.so" in maps


def wait_for_window(config: ResetConfig, trace: dict[str, object], started: float) -> WindowControl:
    deadline = time.monotonic() + config.game_ready_timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        control = WindowControl(
            title=config.window_title,
            input_backend=config.startup_input_backend,
        )
        try:
            control.move_resize(
                left=config.window_left,
                top=config.window_top,
                width=config.window_width,
                height=config.window_height,
            )
            geometry = control.geometry()
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.5)
            continue
        trace.update(
            {
                "window_id": geometry.window_id,
                "window_left": geometry.x,
                "window_top": geometry.y,
                "window_width": geometry.width,
                "window_height": geometry.height,
                "window_ready_ms": elapsed_ms(started),
            }
        )
        return control
    raise RuntimeError(f"WINDOW_NOT_FOUND: {last_error}")


def kill_existing_game() -> None:
    while True:
        try:
            pid = auto_pid()
        except RuntimeError:
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(pid, signal.SIGKILL)


def validate_playable_frame(frame: np.ndarray, config: ResetConfig) -> None:
    mean = float(frame.mean())
    std = float(frame.std())
    if std <= config.image_std_threshold:
        raise RuntimeError(f"image_std too low after reset: {std:.6f}")
    if not config.image_mean_min <= mean <= config.image_mean_max:
        raise RuntimeError(
            f"image_mean out of playable bounds after reset: {mean:.6f} not in "
            f"[{config.image_mean_min:.6f}, {config.image_mean_max:.6f}]"
        )


def failure_stage(exc: Exception) -> str:
    text = str(exc)
    for stage in ("WINDOW_NOT_FOUND", "MODULES_TIMEOUT", "MENU_", "PLAYERCONTROL_TIMEOUT"):
        if stage in text:
            return text.split(":", maxsplit=1)[0]
    return "RESET_FAILED"


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relaunch and validate an owned Getting Over It save."
    )
    parser.add_argument(
        "--resets",
        type=int,
        default=1,
        help="Number of relaunches to validate.",
    )
    parser.add_argument(
        "--launch-command",
        nargs="+",
        default=("steam", "-applaunch", "240720"),
        help="Command used to launch the game.",
    )
    parser.add_argument(
        "--clean-save-path",
        required=True,
        help="Known clean save/checkpoint path.",
    )
    parser.add_argument("--active-save-path", required=True, help="Runtime save path to replace.")
    parser.add_argument("--game-ready-timeout", type=float, default=45.0)
    parser.add_argument("--image-std-threshold", type=float, default=10.0)
    parser.add_argument("--image-mean-min", type=float, default=5.0)
    parser.add_argument("--image-mean-max", type=float, default=250.0)
    parser.add_argument(
        "--startup-input-backend",
        choices=("auto", "xdotool", "ydotool"),
        default="auto",
    )
    parser.add_argument("--startup-probe-timeout", type=float, default=3.0)
    parser.add_argument("--startup-attempts", type=int, default=6)
    parser.add_argument("--startup-key")
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--window-title", default="Getting Over It")
    parser.add_argument("--window-left", type=int, default=None)
    parser.add_argument("--window-top", type=int, default=None)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--title-click", nargs=2, type=int, default=(850, 210))
    parser.add_argument("--confirm-click", nargs=2, type=int, default=(640, 430))
    parser.add_argument("--calibration-samples", type=int, default=1)
    parser.add_argument("--calibration-interval", type=float, default=0.1)
    parser.add_argument("--layout-window", type=int, default=50)
    parser.add_argument("--layout-eps", type=float, default=0.0015)
    parser.add_argument("--output-dir", default="runs/reset_test")
    parser.add_argument("--sleep-after-reset", type=float, default=2.0)
    parser.add_argument("--save-reset-trace", action="store_true")
    parser.add_argument("--no-kill-existing", action="store_true")
    add_capture_region_args(parser)
    return parser


def config_from_args(args: argparse.Namespace) -> ResetConfig:
    capture_region = capture_region_from_args(args)
    if capture_region is None:
        raise SystemExit("Reset test requires an explicit game capture region.")
    if args.resets < 1 or args.game_ready_timeout <= 0 or args.startup_attempts < 0:
        raise SystemExit(
            "--resets and --game-ready-timeout must be positive; "
            "--startup-attempts cannot be negative."
        )
    return ResetConfig(
        launch_command=tuple(args.launch_command),
        clean_save_path=Path(args.clean_save_path),
        active_save_path=Path(args.active_save_path),
        capture_region=capture_region,
        game_ready_timeout=args.game_ready_timeout,
        image_std_threshold=args.image_std_threshold,
        image_mean_min=args.image_mean_min,
        image_mean_max=args.image_mean_max,
        startup_input_backend=args.startup_input_backend,
        startup_reference_dir=Path(args.reference_dir) if args.reference_dir else None,
        startup_title_click=tuple(args.title_click),
        startup_confirm_click=tuple(args.confirm_click),
        startup_key=args.startup_key,
        startup_attempts=args.startup_attempts,
        startup_probe_timeout=args.startup_probe_timeout,
        window_title=args.window_title,
        window_left=args.window_left,
        window_top=args.window_top,
        window_width=args.window_width,
        window_height=args.window_height,
        kill_existing=not args.no_kill_existing,
        calibration_samples=args.calibration_samples,
        calibration_interval=args.calibration_interval,
        layout_window=args.layout_window,
        layout_eps=args.layout_eps,
    )


def write_artifacts(
    tester: RelaunchResetTester,
    output_dir: Path,
    index: int,
    trace: dict[str, object],
    *,
    success: bool,
    write_trace: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if tester.last_frame is not None:
        suffix = "" if success else "_failure"
        write_gray_png(output_dir / f"reset_{index}{suffix}.png", tester.last_frame)
    for label, frame in tester.startup_frames:
        write_gray_png(output_dir / f"reset_{index}_{label}.png", frame)
    if write_trace:
        (output_dir / f"reset_{index}_trace.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def print_result(index: int, result: ResetResult) -> None:
    trace = result.trace
    print(f"RESET {index}")
    for name in (
        "old_pid",
        "pid",
        "reset_mode",
        "process_ready_ms",
        "modules_ready_ms",
        "window_ready_ms",
        "image_ready_ms",
        "startup_action_sent",
        "startup_state",
        "playercontrol_ready_ms",
        "fast_cursor_addr",
        "image_mean",
        "image_std",
    ):
        print(f"  {name}: {trace.get(name, '')}")


def _find_existing_pid() -> int | None:
    try:
        return auto_pid()
    except RuntimeError:
        return None


def main() -> None:
    args = parse_args_allowing_launch_flags(build_parser())
    config = config_from_args(args)
    tester = RelaunchResetTester(config)
    output_dir = Path(args.output_dir)
    try:
        for index in range(args.resets):
            try:
                result = tester.run_once()
            except ResetFailure as exc:
                write_artifacts(
                    tester,
                    output_dir,
                    index,
                    exc.trace,
                    success=False,
                    write_trace=args.save_reset_trace,
                )
                print(f"RESET {index} FAILED")
                print(f"  failure_stage: {exc.trace.get('failure_stage', '')}")
                print(f"  reason: {exc}")
                raise SystemExit(1) from exc
            write_artifacts(
                tester,
                output_dir,
                index,
                result.trace,
                success=True,
                write_trace=args.save_reset_trace,
            )
            print_result(index, result)
            time.sleep(args.sleep_after_reset)
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
