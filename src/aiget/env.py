#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from threading import Event, Lock, Thread

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action_sender import ActionSender, open_action_sender
from .frame_capture import CaptureRegion, FrameCapture
from .live_layout import (
    ResolvedLiveLayout,
    default_live_layout_cache_path,
    load_live_layout,
    resolve_live_layout,
    save_live_layout,
)
from .live_position import freeze_fast_cursor_lane, log
from .memory_probe import MemReader, auto_pid
from .observation_state import (
    PositionSample,
    ProgressTracker,
    RawRichLane,
    SlowLaneAccumulator,
    build_rich_state_snapshot_from_raw,
    empty_live_layout,
    read_rich_raw_sample,
)
from .observation_vector import (
    OBS_DIM,
    RICH_DIM,
    RICH_VALUE_FIELDS,
    build_observation_vector,
)
from .progress_signal import ProgressEstimator
from .reset_startup import StartupAutomation
from .reward import HeightReward
from .window_control import WindowControl

RICH_INDEX = {name: index for index, name in enumerate(RICH_VALUE_FIELDS)}
STATE_OBS_KEY = "state"
IMAGE_OBS_KEY = "image"
IMAGE_HEIGHT = 84
IMAGE_WIDTH = 84
IMAGE_CHANNELS = 4
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
CAPTURE_FRAME_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
RESET_ATTACH = "attach"
RESET_RELAUNCH = "relaunch"


class GettingOverItEnv(gym.Env):
    metadata = {"render_modes": []}

    observation_space = spaces.Dict(
        {
            STATE_OBS_KEY: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(OBS_DIM,),
                dtype=np.float32,
            ),
            IMAGE_OBS_KEY: spaces.Box(
                low=0,
                high=255,
                shape=IMAGE_SHAPE,
                dtype=np.uint8,
            ),
        }
    )
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def __init__(
        self,
        *,
        pid: int | None = None,
        dt: float = 1.0 / 30.0,
        rich_snapshot_interval: float = 0.2,
        action_repeat: int = 2,
        max_episode_seconds: float = 120.0,
        no_progress_timeout: float = 20.0,
        calibration_samples: int = 1,
        calibration_interval: float = 0.7,
        window: int = 0x200,
        eps: float = 0.0015,
        layout_discovery_timeout: float = 1.5,
        live_layout_cache: str | None = None,
        use_layout_cache: bool = True,
        discover_rich_layout: bool = False,
        launch_command: Sequence[str] | None = None,
        reset_backend: str = RESET_ATTACH,
        clean_save_path: str | None = None,
        active_save_path: str | None = None,
        game_ready_timeout: float = 30.0,
        kill_existing_on_reset: bool = True,
        action_sender: ActionSender | None = None,
        enable_uinput: bool = True,
        uinput_device: str = "/dev/uinput",
        max_dx: int = 40,
        max_dy: int = 40,
        capture_region: CaptureRegion | None = None,
        enable_image: bool = True,
        image_hz: float = 30.0,
        strict_image: bool = False,
        image_ready_timeout: float = 3.0,
        image_max_age: float = 0.25,
        image_std_threshold: float = 2.0,
        image_mean_min: float = 0.0,
        image_mean_max: float = 255.0,
        startup_click: tuple[int, int] | None = None,
        startup_key: str | None = None,
        startup_delay: float = 0.0,
        startup_attempts: int = 0,
        startup_mode: str = "legacy",
        startup_reference_dir: str | None = None,
        startup_title_click: tuple[int, int] = (850, 210),
        startup_confirm_click: tuple[int, int] = (640, 430),
        window_title: str = "Getting Over It",
        window_left: int | None = None,
        window_top: int | None = None,
        window_width: int | None = None,
        window_height: int | None = None,
        copy_observation: bool = True,
        debug_json: bool = False,
        debug_every_n: int = 30,
    ):
        self.pid = pid
        self.dt = dt
        self.rich_snapshot_interval = rich_snapshot_interval
        self.action_repeat = max(1, int(action_repeat))
        self.max_episode_seconds = max_episode_seconds
        self.no_progress_timeout = no_progress_timeout
        self.calibration_samples = calibration_samples
        self.calibration_interval = calibration_interval
        self.window = window
        self.eps = eps
        self.layout_discovery_timeout = layout_discovery_timeout
        self.live_layout_cache = live_layout_cache
        self.use_layout_cache = use_layout_cache
        self.discover_rich_layout = discover_rich_layout
        self.launch_command = tuple(launch_command) if launch_command is not None else None
        if reset_backend not in (RESET_ATTACH, RESET_RELAUNCH):
            raise ValueError(f"reset_backend must be 'attach' or 'relaunch', got {reset_backend!r}")
        self.reset_backend = reset_backend
        self.clean_save_path = clean_save_path
        self.active_save_path = active_save_path
        self.game_ready_timeout = game_ready_timeout
        self.kill_existing_on_reset = kill_existing_on_reset
        self._provided_action_sender = action_sender
        self.enable_uinput = enable_uinput
        self.uinput_device = uinput_device
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.capture_region = capture_region
        self.enable_image = enable_image
        self.image_hz = image_hz
        self.strict_image = strict_image
        self.image_ready_timeout = image_ready_timeout
        self.image_max_age = image_max_age
        self.image_std_threshold = image_std_threshold
        self.image_mean_min = image_mean_min
        self.image_mean_max = image_mean_max
        self.startup_click = startup_click
        self.startup_key = startup_key
        self.startup_delay = startup_delay
        self.startup_attempts = max(0, int(startup_attempts))
        if startup_mode not in ("legacy", "auto"):
            raise ValueError(f"startup_mode must be 'legacy' or 'auto', got {startup_mode!r}")
        self.startup_mode = startup_mode
        self.startup_reference_dir = startup_reference_dir
        self.startup_title_click = startup_title_click
        self.startup_confirm_click = startup_confirm_click
        self.window_title = window_title
        self.window_left = window_left
        self.window_top = window_top
        self.window_width = window_width
        self.window_height = window_height
        self.copy_observation = copy_observation
        self.debug_json = debug_json
        self.debug_every_n = max(1, debug_every_n)

        self._layout: ResolvedLiveLayout | None = None
        self._mem_fd = -1
        self._fast_addr = 0
        self._previous_cursor: PositionSample | None = None
        self._previous_action = np.zeros(2, dtype=np.float32)
        self._obs = np.zeros(OBS_DIM, dtype=np.float32)
        self._image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        self._image_latest = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        self._image_frame = np.zeros(CAPTURE_FRAME_SHAPE, dtype=np.uint8)
        self._dict_obs = {STATE_OBS_KEY: self._obs, IMAGE_OBS_KEY: self._image}
        self._image_lock = Lock()
        self._image_stop = Event()
        self._image_thread: Thread | None = None
        self._image_ts = 0.0
        self._image_updates = 0
        self._rich_values = np.zeros(RICH_DIM, dtype=np.float32)
        self._rich_mask = np.zeros(RICH_DIM, dtype=np.float32)
        self._rich_values_local = np.zeros(RICH_DIM, dtype=np.float32)
        self._rich_mask_local = np.zeros(RICH_DIM, dtype=np.float32)
        self._rich_lock = Lock()
        self._rich_stop = Event()
        self._rich_thread: Thread | None = None
        self._rich_ts = 0.0
        self._rich_updates = 0
        self._dropped_rich_updates = 0
        self._rich_accumulator = SlowLaneAccumulator(
            tracker=ProgressTracker(best_height=0.0, last_progress_ts=0.0)
        )
        no_progress_limit_steps = max(1, int(round(self.no_progress_timeout / self.dt)))
        self._progress_estimator = ProgressEstimator()
        self._reward_fn = HeightReward(no_progress_limit_steps=no_progress_limit_steps)
        self._last_reward_debug: dict[str, object] = {}
        self._step_count = 0
        self._episode_started_monotonic = 0.0
        self._game_freeze_detected = False
        self._process_lost = False
        self._last_reset_mode = "not_reset"
        self._reset_started_perf = 0.0
        self._reset_trace: dict[str, object] = {}
        self._startup_frames: list[tuple[str, np.ndarray]] = []
        self._last_best_progress_y = 0.0
        self._action_sender: ActionSender | None = None
        self._frame_capture: FrameCapture | None = None
        self._next_deadline = 0.0
        self.last_step_timing: dict[str, float | int] = {
            "active_step_ms": 0.0,
            "wall_step_ms": 0.0,
            "sleep_ms": 0.0,
            "missed_deadlines": 0,
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_started_perf = time.perf_counter()
        self._reset_trace = {
            "failure_stage": "",
            "reason": "",
            "startup_action_sent": False,
            "startup_actions_sent": 0,
            "old_pid": self.pid,
        }
        self._startup_frames = []
        try:
            return self._reset_impl()
        except Exception as exc:
            self._reset_trace["reason"] = str(exc)
            if not self._reset_trace.get("failure_stage"):
                self._reset_trace["failure_stage"] = "unknown"
            self.close()
            if self.reset_backend == RESET_RELAUNCH and self.kill_existing_on_reset:
                self._kill_existing_game()
            raise

    def _reset_impl(self):
        self.close()
        self._reset_trace["failure_stage"] = "PROCESS_TIMEOUT"
        self.pid = self._reset_game_process()
        self._reset_trace["new_pid"] = self.pid
        self._reset_runtime_state()
        self._open_action_sender()
        if self._auto_startup_enabled():
            self._reset_trace["failure_stage"] = "WINDOW_NOT_FOUND"
            self._prepare_startup_window()
        self._start_frame_capture()
        if self._auto_startup_enabled():
            self._append_startup_full_frame("initial_full")
        if self.enable_image and self.strict_image:
            self._reset_trace["failure_stage"] = "IMAGE_DARK_TIMEOUT"
            self._wait_for_valid_image()
        if self._auto_startup_enabled():
            self._append_startup_full_frame("after_window_focus_full")
        if self._auto_startup_enabled():
            self._layout = self._drive_startup_until_playable(self.pid)
        else:
            self._reset_trace["failure_stage"] = "PLAYERCONTROL_TIMEOUT"
            self._layout = self._load_or_resolve_layout_after_reset(self.pid)
        self._fast_addr = self._layout.fast_cursor_addr
        self._reset_trace["fast_cursor_addr"] = hex(self._fast_addr)
        self._mem_fd = os.open(f"/proc/{self.pid}/mem", os.O_RDONLY)
        self._read_rich_once()
        self._start_rich_thread()
        if self.enable_image and self.strict_image:
            self._wait_for_valid_image()
            self._reset_trace["gameplay_image_ready_ms"] = self._reset_elapsed_ms()
        obs = self.read_observation()
        self._last_best_progress_y = float(obs[STATE_OBS_KEY][13])
        state = obs[STATE_OBS_KEY]
        image = obs[IMAGE_OBS_KEY]
        initial_progress = self._progress_estimator.estimate(state, image)
        self._reward_fn.reset(initial_progress)
        self._last_reward_debug = {
            "initial_progress_y": initial_progress.y,
            "initial_progress_valid": initial_progress.valid,
            "initial_progress_source": initial_progress.source.value,
        }
        self._reset_trace["failure_stage"] = ""
        return self._maybe_copy(obs), self._info()

    def _reset_runtime_state(self) -> None:
        self._previous_cursor = None
        self._previous_action.fill(0.0)
        self._rich_values.fill(0.0)
        self._rich_mask.fill(0.0)
        self._rich_stop = Event()
        self._image_stop = Event()
        self._rich_ts = 0.0
        self._rich_updates = 0
        self._dropped_rich_updates = 0
        self._image.fill(0)
        self._image_latest.fill(0)
        self._image_frame.fill(0)
        self._image_ts = 0.0
        self._image_updates = 0
        self._rich_accumulator = SlowLaneAccumulator(
            tracker=ProgressTracker(best_height=0.0, last_progress_ts=0.0)
        )
        self._step_count = 0
        self._episode_started_monotonic = time.monotonic()
        self._game_freeze_detected = False
        self._process_lost = False
        self._last_best_progress_y = 0.0
        self._reward_fn.reset()
        self._last_reward_debug = {}
        self._next_deadline = time.perf_counter()
        self.last_step_timing = {
            "active_step_ms": 0.0,
            "wall_step_ms": 0.0,
            "sleep_ms": 0.0,
            "missed_deadlines": 0,
        }

    def _auto_startup_enabled(self) -> bool:
        return self.reset_backend == RESET_RELAUNCH and self.startup_mode == "auto"

    @property
    def startup_frames(self) -> list[tuple[str, np.ndarray]]:
        return [(label, frame.copy()) for label, frame in self._startup_frames]

    def _open_action_sender(self) -> None:
        self._action_sender = self._provided_action_sender or open_action_sender(
            enabled=self.enable_uinput,
            device=self.uinput_device,
            max_dx=self.max_dx,
            max_dy=self.max_dy,
        )

    def _start_frame_capture(self) -> None:
        self._frame_capture = (
            FrameCapture(
                output_shape=CAPTURE_FRAME_SHAPE,
                region=self.capture_region,
                allow_blank=not self.strict_image,
                prefer_xwd=self._auto_startup_enabled(),
            )
            if self.enable_image
            else None
        )
        self._start_image_thread()

    def step(self, action):
        action_array = np.asarray(action, dtype=np.float32)
        action_array = np.clip(action_array, -1.0, 1.0)
        wall_started = time.perf_counter()
        total_reward = 0.0
        total_active = 0.0
        total_sleep = 0.0
        missed_deadlines = 0
        obs = self._dict_obs
        terminated = False
        truncated = False

        for _ in range(self.action_repeat):
            active_started = time.perf_counter()
            try:
                if self._action_sender is not None:
                    self._action_sender.send_mouse_delta(
                        (float(action_array[0]), float(action_array[1]))
                    )
                obs = self.read_observation()
                reward, terminated, truncated = self._compute_reward_and_done(obs)
            except ProcessLookupError:
                reward = 0.0
                terminated = True
                truncated = False
                self._process_lost = True
            total_reward += reward
            total_active += time.perf_counter() - active_started

            self._next_deadline += self.dt
            sleep_time = self._next_deadline - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
                total_sleep += sleep_time
            else:
                missed_deadlines += 1
                self._next_deadline = time.perf_counter()
            if terminated or truncated:
                break

        wall_elapsed = time.perf_counter() - wall_started
        self._game_freeze_detected = self._game_freeze_detected or wall_elapsed > max(
            0.25,
            self.dt * self.action_repeat * 4.0,
        )
        self._previous_action[:] = action_array[:2]
        self._step_count += 1
        self.last_step_timing = {
            "active_step_ms": total_active * 1000.0,
            "wall_step_ms": wall_elapsed * 1000.0,
            "sleep_ms": total_sleep * 1000.0,
            "missed_deadlines": missed_deadlines,
        }
        if self.debug_json and self._step_count % self.debug_every_n == 0:
            print(json.dumps(self._debug_payload(obs[STATE_OBS_KEY], total_reward)), flush=True)
        return self._maybe_copy(obs), total_reward, terminated, truncated, self._info()

    def read_observation(self) -> dict[str, np.ndarray]:
        self.read_observation_vector()
        with self._image_lock:
            self._image[:] = self._image_latest
        return self._dict_obs

    def read_observation_vector(self) -> np.ndarray:
        try:
            data = os.pread(self._mem_fd, 8, self._fast_addr)
        except OSError as exc:
            raise ProcessLookupError("cursor read failed; game process likely died") from exc
        if len(data) != 8:
            raise ProcessLookupError("short cursor read; game process likely died")
        cursor_x, cursor_y = np.frombuffer(data, dtype="<f4", count=2)
        now = time.time()
        cursor = PositionSample(ts=now, x=float(cursor_x), y=float(cursor_y))
        if self._previous_cursor is None or cursor.ts <= self._previous_cursor.ts:
            cursor_vxy = (0.0, 0.0)
        else:
            dt = cursor.ts - self._previous_cursor.ts
            cursor_vxy = (
                (cursor.x - self._previous_cursor.x) / dt,
                (cursor.y - self._previous_cursor.y) / dt,
            )
        self._previous_cursor = cursor

        with self._rich_lock:
            self._rich_values_local[:] = self._rich_values
            self._rich_mask_local[:] = self._rich_mask

        return build_observation_vector(
            (cursor.x, cursor.y),
            cursor_vxy,
            (self._rich_values_local, self._rich_mask_local),
            self._previous_action,
            self._obs,
        )

    def close(self) -> None:
        self._rich_stop.set()
        self._image_stop.set()
        if self._rich_thread is not None:
            self._rich_thread.join(timeout=1.0)
            self._rich_thread = None
        if self._image_thread is not None:
            image_thread = self._image_thread
            image_thread.join(timeout=1.0)
            if image_thread.is_alive():
                self._frame_capture = None
            self._image_thread = None
        if self._frame_capture is not None:
            self._frame_capture.close()
            self._frame_capture = None
        if self._mem_fd >= 0:
            os.close(self._mem_fd)
            self._mem_fd = -1
        if (
            self._action_sender is not None
            and self._action_sender is not self._provided_action_sender
        ):
            self._action_sender.close()
        self._action_sender = None

    def _reset_game_process(self) -> int:
        if self.reset_backend == RESET_ATTACH:
            self._last_reset_mode = "attach_no_game_reset"
            return self.pid or self._launch_or_attach_game()

        if self.launch_command is None:
            raise RuntimeError("reset_backend='relaunch' requires launch_command")
        if self.clean_save_path is None or self.active_save_path is None:
            raise RuntimeError(
                "reset_backend='relaunch' requires clean_save_path and active_save_path"
            )
        try:
            self._reset_trace["old_pid"] = auto_pid()
        except RuntimeError:
            self._reset_trace["old_pid"] = self.pid
        if self.kill_existing_on_reset:
            self._kill_existing_game()
        self.pid = None
        self._restore_save()
        subprocess.Popen(self.launch_command)
        self._last_reset_mode = "relaunch_save_restore"
        return self._wait_for_game_process()

    def cleanup_reset_processes(self) -> None:
        self.close()
        if self.kill_existing_on_reset:
            self._kill_existing_game()

    @property
    def reset_trace(self) -> dict[str, object]:
        return dict(self._reset_trace)

    def _launch_or_attach_game(self) -> int:
        try:
            return auto_pid()
        except RuntimeError:
            if self.launch_command is None:
                raise
            subprocess.Popen(self.launch_command)
            return self._wait_for_game_process()

    def _wait_for_game_process(self) -> int:
        deadline = time.monotonic() + self.game_ready_timeout
        process_seen = False
        while time.monotonic() < deadline:
            try:
                pid = auto_pid()
            except RuntimeError:
                time.sleep(0.5)
                continue
            if not process_seen:
                self._reset_trace["process_ready_ms"] = self._reset_elapsed_ms()
                process_seen = True
            self._reset_trace["failure_stage"] = "MODULES_TIMEOUT"
            if self._game_process_ready(pid):
                self._reset_trace["modules_ready_ms"] = self._reset_elapsed_ms()
                return pid
            time.sleep(0.5)
        raise RuntimeError("GettingOverIt.x86_64 did not appear after launch_command") from None

    def _game_process_ready(self, pid: int) -> bool:
        try:
            maps = Path(f"/proc/{pid}/maps").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return "GameAssembly.so" in maps and "UnityPlayer.so" in maps

    def _kill_existing_game(self) -> None:
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

    def _restore_save(self) -> None:
        assert self.clean_save_path is not None
        assert self.active_save_path is not None
        clean = Path(self.clean_save_path).expanduser()
        active = Path(self.active_save_path).expanduser()
        if not clean.exists():
            raise RuntimeError(f"clean save path does not exist: {clean}")
        active.parent.mkdir(parents=True, exist_ok=True)
        if clean.is_dir():
            if active.exists() and not active.is_dir():
                active.unlink()
            if active.exists():
                shutil.rmtree(active)
            shutil.copytree(clean, active)
        else:
            if active.exists() and active.is_dir():
                shutil.rmtree(active)
            shutil.copy2(clean, active)

    def _load_or_resolve_layout(self, pid: int) -> ResolvedLiveLayout:
        if self.use_layout_cache:
            cache_path = self.live_layout_cache or default_live_layout_cache_path(pid)
        else:
            cache_path = None
        if cache_path:
            try:
                cached_layout = load_live_layout(cache_path)
            except FileNotFoundError:
                pass
            else:
                if cached_layout.pid == pid and (
                    not self.discover_rich_layout or _layout_has_progress_fields(cached_layout)
                ):
                    return cached_layout

        fast_lane = freeze_fast_cursor_lane(
            pid=pid,
            calibration_samples=self.calibration_samples,
            calibration_interval=self.calibration_interval,
            window=self.window,
            eps=self.eps,
        )
        if not self.discover_rich_layout:
            layout = empty_live_layout(pid, fast_lane.current_addr)
        else:
            try:
                layout = resolve_live_layout(
                    pid,
                    calibration_samples=self.calibration_samples,
                    calibration_interval=self.calibration_interval,
                    window=self.window,
                    eps=self.eps,
                    startup_timeout=self.layout_discovery_timeout,
                    resolve_optional_fields=False,
                    fast_cursor_addr=fast_lane.current_addr,
                )
            except Exception as exc:
                log(
                    "Rich raw layout discovery failed fast; "
                    f"falling back to fast-lane-only env: {exc}"
                )
                layout = empty_live_layout(pid, fast_lane.current_addr)
        if cache_path:
            save_live_layout(cache_path, layout)
        return layout

    def _load_or_resolve_layout_after_reset(self, pid: int) -> ResolvedLiveLayout:
        if self.reset_backend != RESET_RELAUNCH:
            return self._load_or_resolve_layout(pid)

        deadline = time.monotonic() + self.game_ready_timeout
        last_exc: Exception | None = None
        next_startup_action_at = time.monotonic() + max(0.0, self.startup_delay)
        while time.monotonic() < deadline:
            refreshed_pid = self._refresh_reset_pid_if_needed(pid)
            if refreshed_pid is None:
                time.sleep(0.5)
                continue
            pid = refreshed_pid
            if (
                self._startup_action_configured()
                and int(self._reset_trace["startup_actions_sent"]) < self.startup_attempts
                and time.monotonic() >= next_startup_action_at
            ):
                self._send_startup_action()
                next_startup_action_at = time.monotonic() + max(0.5, self.startup_delay)
            try:
                layout = self._load_or_resolve_layout(pid)
                self._reset_trace["playercontrol_ready_ms"] = self._reset_elapsed_ms()
                return layout
            except Exception as exc:
                last_exc = exc
                log(f"Relaunch layout discovery not ready yet; retrying: {exc}")
                time.sleep(1.0)
        raise RuntimeError("layout discovery did not become ready after relaunch") from last_exc

    def _prepare_startup_window(self) -> None:
        deadline = time.monotonic() + self.game_ready_timeout
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            control = WindowControl(title=self.window_title)
            try:
                control.move_resize(
                    left=self.window_left,
                    top=self.window_top,
                    width=self.window_width,
                    height=self.window_height,
                )
                geom = control.geometry()
            except Exception as exc:
                last_exc = exc
                time.sleep(0.5)
                continue
            self._reset_trace["window_id"] = geom.window_id
            self._reset_trace["window_left"] = geom.x
            self._reset_trace["window_top"] = geom.y
            self._reset_trace["window_width"] = geom.width
            self._reset_trace["window_height"] = geom.height
            self._reset_trace["window_ready_ms"] = self._reset_elapsed_ms()
            return
        raise RuntimeError(f"WINDOW_NOT_FOUND: {last_exc}") from last_exc

    def _drive_startup_until_playable(self, pid: int) -> ResolvedLiveLayout:
        self._reset_trace["failure_stage"] = "PLAYERCONTROL_TIMEOUT"
        control = WindowControl(title=self.window_title)

        def resolve_layout() -> ResolvedLiveLayout:
            refreshed_pid = self._refresh_reset_pid_if_needed(pid)
            if refreshed_pid is None:
                raise RuntimeError("PROCESS_TIMEOUT")
            return self._load_or_resolve_layout(refreshed_pid)

        automation = StartupAutomation(
            frame_getter=self._latest_image_frame,
            full_frame_getter=self._capture_startup_full_frame,
            window_control=control,
            resolve_layout=resolve_layout,
            timeout=self.game_ready_timeout,
            reference_dir=self.startup_reference_dir,
            action_interval=2.0,
            max_actions=self.startup_attempts or 6,
            title_click=self.startup_title_click,
            confirm_click=self.startup_confirm_click,
        )
        result = automation.drive_until_playable()
        startup_frames = list(self._startup_frames)
        for event in result.events:
            if event.action == "resolved":
                startup_frames.append(("playable_full", event.frame.copy()))
                continue
            startup_frames.append((f"before_click_{event.index}_full", event.frame.copy()))
            if event.after_frame is not None:
                startup_frames.append((f"after_click_{event.index}_full", event.after_frame.copy()))
        self._startup_frames = startup_frames
        self._reset_trace["startup"] = result.as_trace()
        self._reset_trace["startup_state"] = result.state.value
        self._reset_trace["startup_actions_sent"] = result.attempts
        self._reset_trace["startup_action_sent"] = result.attempts > 0
        if not result.success or result.layout is None:
            reason = result.reason or "MENU_UNKNOWN_TIMEOUT"
            self._reset_trace["failure_stage"] = reason
            self._append_startup_full_frame("failure_full")
            raise RuntimeError(f"{reason}: startup did not reach playable scene")
        self._reset_trace["playercontrol_ready_ms"] = self._reset_elapsed_ms()
        return result.layout

    def _latest_image_frame(self) -> np.ndarray:
        with self._image_lock:
            return self._image_latest.copy()

    def _capture_startup_full_frame(self) -> np.ndarray:
        if self.capture_region is not None:
            region = self.capture_region
            output_shape = (region.height, region.width, 1)
        else:
            output_shape = (self.window_height or 720, self.window_width or 1280, 1)
            region = None
        capture = FrameCapture(
            output_shape=output_shape,
            region=region,
            allow_blank=False,
            prefer_xwd=True,
        )
        try:
            return capture.read_full_gray()
        finally:
            capture.close()

    def _append_startup_full_frame(self, label: str) -> None:
        try:
            frame = self._capture_startup_full_frame()
        except Exception as exc:
            self._reset_trace[f"{label}_error"] = str(exc)
            return
        self._startup_frames.append((label, frame))

    def _refresh_reset_pid_if_needed(self, pid: int) -> int | None:
        if Path(f"/proc/{pid}/maps").exists():
            return pid
        try:
            new_pid = auto_pid()
        except RuntimeError:
            return None
        if not self._game_process_ready(new_pid):
            return None
        self.pid = new_pid
        self._reset_trace["new_pid"] = new_pid
        self._reset_trace["reacquired_pid"] = new_pid
        return new_pid

    def _startup_action_configured(self) -> bool:
        return self.startup_attempts > 0 and (
            self.startup_click is not None or self.startup_key is not None
        )

    def _send_startup_action(self) -> None:
        commands: list[list[str]] = []
        window_info = self._game_window_info()
        window_id = window_info.get("WINDOW") if window_info is not None else None
        if window_id:
            try:
                subprocess.run(
                    ["xdotool", "windowactivate", "--sync", window_id],
                    check=False,
                    timeout=2.0,
                )
            except Exception:
                pass
        if self.startup_click is not None:
            x, y = self.startup_click
            if window_info is not None and "X" in window_info and "Y" in window_info:
                x += int(window_info["X"])
                y += int(window_info["Y"])
            elif self.capture_region is not None:
                x += self.capture_region.left
                y += self.capture_region.top
            commands.append(["xdotool", "mousemove", str(x), str(y), "click", "1"])
        if self.startup_key is not None:
            if window_id:
                commands.append(["xdotool", "key", "--window", window_id, self.startup_key])
            else:
                commands.append(["xdotool", "key", self.startup_key])
        for command in commands:
            try:
                subprocess.run(command, check=True, timeout=3.0)
            except Exception as exc:
                raise RuntimeError(f"startup action failed: {' '.join(command)}: {exc}") from exc
        self._reset_trace["startup_action_sent"] = True
        self._reset_trace["startup_actions_sent"] = (
            int(self._reset_trace.get("startup_actions_sent", 0)) + 1
        )

    def _game_window_info(self) -> dict[str, str] | None:
        try:
            completed = subprocess.run(
                [
                    "xdotool",
                    "search",
                    "--onlyvisible",
                    "--class",
                    "GettingOverIt",
                    "getwindowgeometry",
                    "--shell",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except Exception:
            return None
        values: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            key, sep, value = line.partition("=")
            if sep:
                values[key] = value
        return values or None

    def _reset_elapsed_ms(self) -> float:
        if self._reset_started_perf <= 0.0:
            return 0.0
        return (time.perf_counter() - self._reset_started_perf) * 1000.0

    def _start_rich_thread(self) -> None:
        self._rich_thread = Thread(
            target=self._rich_loop,
            name="aiget-rich-observation",
            daemon=True,
        )
        self._rich_thread.start()

    def _start_image_thread(self) -> None:
        if self._frame_capture is None:
            self._image.fill(0)
            self._image_latest.fill(0)
            self._image_frame.fill(0)
            return
        self._image_thread = Thread(
            target=self._image_loop,
            args=(self._frame_capture,),
            name="aiget-image-observation",
            daemon=True,
        )
        self._image_thread.start()

    def _image_loop(self, capture: FrameCapture | None) -> None:
        interval = 1.0 / self.image_hz if self.image_hz > 0 else 0.0
        try:
            while not self._image_stop.is_set():
                try:
                    assert capture is not None
                    frame = capture.read()
                    with self._image_lock:
                        self._image_frame[:] = frame
                        self._image_latest[:, :, :-1] = self._image_latest[:, :, 1:]
                        self._image_latest[:, :, -1] = frame[:, :, 0]
                        self._image_ts = time.time()
                        self._image_updates += 1
                except Exception as exc:
                    log(f"Image observation refresh failed: {exc}")
                self._image_stop.wait(interval)
        finally:
            if capture is not None:
                capture.close()

    def _wait_for_valid_image(self) -> None:
        deadline = time.monotonic() + self.image_ready_timeout
        while time.monotonic() < deadline:
            with self._image_lock:
                updates = self._image_updates
                image_age = 0.0 if self._image_ts <= 0.0 else time.time() - self._image_ts
                image_mean = float(self._image_latest.mean())
                image_std = float(self._image_latest.std())
            if (
                updates > 0
                and image_age <= self.image_max_age
                and image_std > self.image_std_threshold
                and self.image_mean_min <= image_mean <= self.image_mean_max
            ):
                self._reset_trace["image_ready_ms"] = self._reset_elapsed_ms()
                return
            time.sleep(0.02)
        with self._image_lock:
            updates = self._image_updates
            image_age = 0.0 if self._image_ts <= 0.0 else time.time() - self._image_ts
            image_mean = float(self._image_latest.mean())
            image_std = float(self._image_latest.std())
        raise RuntimeError(
            "image capture did not become valid: "
            f"updates={updates} age={image_age:.3f}s "
            f"mean={image_mean:.3f} std={image_std:.3f}"
        )

    def _rich_loop(self) -> None:
        while not self._rich_stop.is_set():
            try:
                self._read_rich_once()
            except Exception as exc:
                log(f"Rich observation refresh failed: {exc}")
                self._dropped_rich_updates += 1
            self._rich_stop.wait(max(0.0, self.rich_snapshot_interval))

    def _read_rich_once(self) -> None:
        assert self._layout is not None
        lane = RawRichLane(
            pid=self.pid or self._layout.pid,
            refresh_interval=self.rich_snapshot_interval,
            layout=self._layout,
        )
        with MemReader(lane.pid) as reader:
            sample = read_rich_raw_sample(reader, lane)
        if sample.progress_features is not None:
            self._rich_accumulator.tracker.best_height = max(
                self._rich_accumulator.tracker.best_height,
                self._last_best_progress_y,
                sample.progress_features[0],
            )
        rich_state = build_rich_state_snapshot_from_raw(sample, self._rich_accumulator)
        values, mask = self._rich_state_to_arrays(rich_state)
        with self._rich_lock:
            self._rich_values[:] = values
            self._rich_mask[:] = mask
            self._rich_ts = rich_state.ts
            self._rich_updates += 1

    def _rich_state_to_arrays(self, rich_state) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros(RICH_DIM, dtype=np.float32)
        mask = np.zeros(RICH_DIM, dtype=np.float32)
        valid = rich_state.valid_mask

        body_valid = 1.0 if valid.get("body_position_xy", False) else 0.0
        hammer_tip_valid = 1.0 if valid.get("hammer_tip_xy", False) else 0.0
        hammer_dir_valid = (
            1.0
            if valid.get("hammer_anchor_xy", False) and valid.get("hammer_tip_xy", False)
            else 0.0
        )
        progress_valid = 1.0 if valid.get("progress_features", False) else 0.0

        values[RICH_INDEX["body_x"]] = rich_state.body_position_xy[0]
        values[RICH_INDEX["body_y"]] = rich_state.body_position_xy[1]
        values[RICH_INDEX["body_vx"]] = rich_state.body_velocity_xy[0]
        values[RICH_INDEX["body_vy"]] = rich_state.body_velocity_xy[1]
        values[RICH_INDEX["hammer_tip_x"]] = rich_state.hammer_tip_xy[0]
        values[RICH_INDEX["hammer_tip_y"]] = rich_state.hammer_tip_xy[1]
        values[RICH_INDEX["hammer_dir_sin"]] = rich_state.hammer_direction_sin_cos[0]
        values[RICH_INDEX["hammer_dir_cos"]] = rich_state.hammer_direction_sin_cos[1]
        values[RICH_INDEX["progress_y"]] = rich_state.progress_features[0]
        values[RICH_INDEX["best_progress_y"]] = rich_state.progress_features[1]
        values[RICH_INDEX["time_since_progress"]] = rich_state.progress_features[2]

        mask[RICH_INDEX["body_x"]] = body_valid
        mask[RICH_INDEX["body_y"]] = body_valid
        mask[RICH_INDEX["body_vx"]] = body_valid
        mask[RICH_INDEX["body_vy"]] = body_valid
        mask[RICH_INDEX["hammer_tip_x"]] = hammer_tip_valid
        mask[RICH_INDEX["hammer_tip_y"]] = hammer_tip_valid
        mask[RICH_INDEX["hammer_dir_sin"]] = hammer_dir_valid
        mask[RICH_INDEX["hammer_dir_cos"]] = hammer_dir_valid
        mask[RICH_INDEX["progress_y"]] = progress_valid
        mask[RICH_INDEX["best_progress_y"]] = progress_valid
        mask[RICH_INDEX["time_since_progress"]] = progress_valid
        return values, mask

    def _compute_reward_and_done(self, obs_dict: dict[str, np.ndarray]) -> tuple[float, bool, bool]:
        state = obs_dict[STATE_OBS_KEY]
        image = obs_dict[IMAGE_OBS_KEY]

        progress = self._progress_estimator.estimate(state, image)

        out = self._reward_fn.compute(
            progress=progress,
            time_limit_reached=self._time_limit_reached(time.monotonic()),
        )

        self._last_reward_debug = out.debug
        return out.reward, out.terminated, out.truncated

    def _time_limit_reached(self, now: float) -> bool:
        return now - self._episode_started_monotonic >= self.max_episode_seconds

    def _info(self) -> dict[str, object]:
        rich_age = 0.0 if self._rich_ts <= 0.0 else max(0.0, time.time() - self._rich_ts)
        reward_debug = dict(self._last_reward_debug)
        return {
            "pid": self.pid,
            "fast_addr": hex(self._fast_addr),
            "rich_state_age": rich_age,
            "rich_updates": self._rich_updates,
            "dropped_rich_updates": self._dropped_rich_updates,
            "image_updates": self._image_updates,
            "image_age": 0.0 if self._image_ts <= 0.0 else max(0.0, time.time() - self._image_ts),
            "image_mean": float(self._image_latest.mean()),
            "image_std": float(self._image_latest.std()),
            "image_min": int(self._image_latest.min()),
            "image_max": int(self._image_latest.max()),
            "game_freeze_detected": self._game_freeze_detected,
            "process_lost": self._process_lost,
            "step_timing": dict(self.last_step_timing),
            "reset_mode": self._last_reset_mode,
            "reset_trace": dict(self._reset_trace),
            "progress_y": reward_debug.get(
                "progress_y", reward_debug.get("initial_progress_y", 0.0)
            ),
            "progress_valid": reward_debug.get(
                "progress_valid", reward_debug.get("initial_progress_valid", False)
            ),
            "progress_source": reward_debug.get(
                "progress_source", reward_debug.get("initial_progress_source", "invalid")
            ),
            "reward_reason": reward_debug.get("reward_reason", "missing"),
            "reward_debug": reward_debug,
        }

    def _debug_payload(self, obs: np.ndarray, reward: float) -> dict[str, object]:
        return {
            "step": self._step_count,
            "reward": reward,
            "cursor": obs[0:4].tolist(),
            "rich": obs[4:15].tolist(),
            "mask": obs[17:].tolist(),
            "info": self._info(),
        }

    def _maybe_copy(self, obs):
        if not self.copy_observation:
            return obs
        if isinstance(obs, dict):
            return {key: value.copy() for key, value in obs.items()}
        return obs.copy()


def _layout_has_progress_fields(layout: ResolvedLiveLayout) -> bool:
    return bool(
        layout.body_position_addr is not None
        and layout.progress_addr is not None
        and layout.valid_mask.get("body_position_xy")
        and layout.valid_mask.get("progress_features")
    )
