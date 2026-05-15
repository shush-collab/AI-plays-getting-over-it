# AIget Environment

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-native-FCC624?style=flat-square&logo=linux&logoColor=black)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environment-0081A5?style=flat-square)
![Unity](https://img.shields.io/badge/Unity-IL2CPP%20Memory-000000?style=flat-square&logo=unity&logoColor=white)
![Status](https://img.shields.io/badge/Status-Environment%20Infrastructure-blueviolet?style=flat-square)

A native Linux reinforcement learning **environment and control stack** for *Getting Over It with Bennett Foddy*.

This repository focuses on the infrastructure needed before learning can work:

- reading useful game state from a native Unity/IL2CPP process
- sending low-latency mouse actions through Linux
- exposing a Gymnasium-compatible environment
- creating deterministic reset/startup behavior
- validating reward and progress signals
- benchmarking observation/control latency

The actual **agent learning code, model experiments, checkpoints, training curves, policies, and evaluation videos** are intentionally kept separate from this repository.

---

## Scope

This repo is the **environment layer**, not the final AI agent.

```text
AIget Environment
├── Game process control
├── Linux mouse action sender
├── Unity/IL2CPP memory observation
├── Image observation
├── Gymnasium environment API
├── Reset automation
├── Reward/progress signal debugging
└── Smoke tests and validation tools

Separate agent/training layer
├── RL algorithms
├── policy architectures
├── experiment tracking
├── checkpoints
├── training curves
└── evaluation videos
```

This separation keeps the project clean:

- environment code stays reproducible and testable
- agent experiments can change freely without breaking the environment
- reward bugs, reset bugs, and learning bugs can be debugged independently

---

## Why This Is Hard

Most game RL projects use a clean simulator API.

*Getting Over It* does not provide one.

This project builds an environment around an existing native Linux game by combining:

- raw process memory reads
- Unity/IL2CPP object discovery
- live observation caching
- screen capture
- Linux input injection
- deterministic save restore
- menu/startup automation
- reward validation

The main challenge is not only “train an agent.”  
The first challenge is building a stable environment where an agent can safely learn.

---

## Architecture

```text
Native Linux Game
      │
      ├── Unity/IL2CPP memory probing
      │        ├── PlayerControl discovery
      │        ├── fakeCursorRB lookup
      │        └── raw memory position reads
      │
      ├── Screen capture
      │        └── 84x84 grayscale frame stack
      │
      ▼
Observation Builder
      ├── 32-float state vector
      ├── image observation
      ├── validity masks
      └── reward/progress diagnostics
      │
      ▼
Gymnasium Environment
      ├── reset()
      ├── step(action)
      ├── Dict observation space
      └── normalized 2D mouse action space
      │
      ▼
Linux Action Sender
      └── /dev/uinput relative mouse movement
```

---

## Current Status

Working:

- Gymnasium-compatible `GettingOverItEnv`
- normalized 2D mouse action space
- Linux `/dev/uinput` mouse control
- stacked image observation
- fixed 32-float state vector
- raw memory fast cursor lane
- partial rich-state memory lane
- reward/progress diagnostics
- deterministic relaunch/save-restore reset on the current setup
- random rollout smoke testing
- reward signal preflight
- environment validation with SB3 `check_env`

Still in progress:

- resolving all rich game-state fields from raw memory
- replacing external rich reads with a cleaner in-game shared-memory observation blob
- improving reward shaping beyond simple height progress
- making reset/menu automation portable across more screen layouts
- separating long-running agent training into its own project/layer

---

## Repository Layout

```text
.
├── docs/
│   ├── observation-schema.md
│   ├── observation-roadmap.md
│   └── player-movement.md
│
├── src/aiget/
│   ├── env.py                    # Gymnasium environment
│   ├── action_sender.py          # Linux /dev/uinput mouse control
│   ├── frame_capture.py          # image capture and frame stack
│   ├── observation_vector.py     # 32-float state vector
│   ├── observation_state.py      # combined live observation state
│   ├── observation_schema.py     # observation layout definition
│   ├── ptrace_il2cpp.py          # Unity/IL2CPP resolver
│   ├── live_position.py          # fast raw-memory cursor lane
│   ├── live_layout.py            # reusable raw memory layout
│   ├── memory_probe.py           # memory inspection helper
│   ├── progress_signal.py        # progress_y extraction
│   ├── reward.py                 # reward calculation
│   ├── debug_reward_signal.py    # live reward preflight
│   ├── benchmark_observation.py  # observation timing benchmark
│   ├── random_rollout.py         # random action smoke rollout
│   ├── check_env.py              # Gymnasium/SB3 env validation
│   └── test_reset.py             # relaunch/save-restore reset test
│
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

Root-level `goi_*.py` files are compatibility wrappers around the package in `src/aiget/`.

---

## Installation

### Using `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv sync
```

For development tools:

```bash
uv sync --extra dev
```

For RL environment validation tools:

```bash
uv sync --extra rl
```

### Using `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"
pip install -e ".[rl]"
```

---

## Basic Usage

```python
from aiget.env import GettingOverItEnv

env = GettingOverItEnv(dt=1.0 / 30.0)

obs, info = env.reset()

state = obs["state"]  # float32, shape=(32,)
image = obs["image"]  # uint8, shape=(84, 84, 4)

obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

env.close()
```

---

## Observation Space

The environment returns a Gymnasium `Dict` observation:

```text
state: float32[32]
image: uint8[84, 84, 4]
```

The state vector contains compact numeric signals such as cursor/body/progress-related values and validity flags.

The image lane gives the agent visual motion context through a 4-frame grayscale stack.

---

## Action Space

The environment accepts normalized 2D mouse actions:

```text
action = [dx, dy]
```

Actions are sent to the game as relative mouse movement through Linux `/dev/uinput`.

---

## Reward Signal

The current reward is intentionally simple and debuggable:

```text
reward = height/progress improvement
```

The reward prefers memory-derived body/progress values. If valid memory progress is unavailable, the sample is marked invalid instead of silently producing misleading reward.

Every `env.step()` exposes reward diagnostics in `info`:

```text
progress_y
progress_valid
progress_source
reward_reason
reward_debug
```

This makes reward bugs visible before long training runs.

---

## Validation Commands

Run repository-level checks:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall src *.py
```

Validate the Gymnasium environment:

```bash
python -m aiget.check_env --allow-attach-reset --steps 100
```

Run an observation benchmark:

```bash
python -m aiget.benchmark_observation --seconds 10
```

Run an attach-mode random-action smoke rollout:

```bash
python -m aiget.random_rollout \
  --seconds 60 \
  --discover-rich-layout \
  --capture-left 320 \
  --capture-top 178 \
  --capture-width 1920 \
  --capture-height 1080 \
  --csv runs/random_rollout.csv
```

Debug attach-mode reward signal quality:

```bash
python -m aiget.debug_reward_signal \
  --seconds 120 \
  --send-actions \
  --capture-left 320 \
  --capture-top 178 \
  --capture-width 1920 \
  --capture-height 1080 \
  --csv runs/reward_signal.csv
```

A useful reward preflight should show:

```text
progress_valid_ratio >= 0.8
reward_std > 0.0001
progress source mostly memory_body or memory_progress
reward reason mostly height_progress
```

---

## Reset Testing

The environment supports a relaunch/save-restore reset path for the current local setup.

Example:

```bash
python -m aiget.test_reset \
  --resets 5 \
  --reset-backend relaunch \
  --launch-command steam -applaunch 240720 -- -screen-fullscreen 0 -screen-width 1920 -screen-height 1080 \
  --clean-save-path "$HOME/goi_reset_saves/start_clean" \
  --active-save-path "$HOME/.config/unity3d/Bennett Foddy/Getting Over It" \
  --startup-mode auto \
  --title-click 1275 305 \
  --confirm-click 1275 305 \
  --window-left 320 --window-top 178 --window-width 1920 --window-height 1080 \
  --capture-left 320 --capture-top 178 --capture-width 1920 --capture-height 1080
```

Current measured geometry:

```text
window/capture: 320,178,1920,1080
title click:    1275,305
confirm click:  1275,305
```

Relaunch reward proof uses the same reset path as training:

```bash
python -m aiget.debug_reward_signal \
  --seconds 300 \
  --send-actions \
  --reset-backend relaunch \
  --launch-command steam -applaunch 240720 -- -screen-fullscreen 0 -screen-width 1920 -screen-height 1080 \
  --clean-save-path "$HOME/goi_reset_saves/start_clean" \
  --active-save-path "$HOME/.config/unity3d/Bennett Foddy/Getting Over It" \
  --startup-mode auto \
  --title-click 1275 305 \
  --confirm-click 1275 305 \
  --window-left 320 --window-top 178 --window-width 1920 --window-height 1080 \
  --capture-left 320 --capture-top 178 --capture-width 1920 --capture-height 1080 \
  --csv runs/reward_signal_relaunch_300s.csv
```

Relaunch random rollout proof:

```bash
python -m aiget.random_rollout \
  --seconds 300 \
  --send-actions \
  --reset-backend relaunch \
  --launch-command steam -applaunch 240720 -- -screen-fullscreen 0 -screen-width 1920 -screen-height 1080 \
  --clean-save-path "$HOME/goi_reset_saves/start_clean" \
  --active-save-path "$HOME/.config/unity3d/Bennett Foddy/Getting Over It" \
  --startup-mode auto \
  --title-click 1275 305 \
  --confirm-click 1275 305 \
  --window-left 320 --window-top 178 --window-width 1920 --window-height 1080 \
  --capture-left 320 --capture-top 178 --capture-width 1920 --capture-height 1080 \
  --strict-image \
  --discover-rich-layout \
  --csv runs/random_rollout_relaunch_300s.csv
```

Relaunch reward and rollout proof should show:

- `reset_mode = relaunch_save_restore`
- `progress_valid_ratio >= 0.8`
- `reward_std > 0.0001`
- progress source mostly `memory_body` or `memory_progress`
- reward reason mostly `height_progress`
- `process_lost = False`

Training refuses attach-only reset by default. Real training should use
`--reset-backend relaunch`, a known clean save, an active save path, a launch
command, an explicit capture region, and reward/progress preflight guards:

```bash
python -m aiget.train_sac \
  --algo sac \
  --steps 10000 \
  --send-actions \
  --reset-backend relaunch \
  --launch-command steam -applaunch 240720 -- -screen-fullscreen 0 -screen-width 1920 -screen-height 1080 \
  --clean-save-path "$HOME/goi_reset_saves/start_clean" \
  --active-save-path "$HOME/.config/unity3d/Bennett Foddy/Getting Over It" \
  --startup-mode auto \
  --title-click 1275 305 \
  --confirm-click 1275 305 \
  --window-left 320 --window-top 178 --window-width 1920 --window-height 1080 \
  --capture-left 320 --capture-top 178 --capture-width 1920 --capture-height 1080 \
  --preflight-steps 300 \
  --min-reward-std 0.0001 \
  --min-progress-valid-ratio 0.8
```

Training writes SB3 Monitor episode diagnostics to `runs/monitor.csv`.

---

## Console Scripts

After installation, these commands are available:

```bash
aiget-live-position --format json
aiget-memory-probe icalls
aiget-observation-schema --format markdown
aiget-observation-state --format json
aiget-benchmark-observation --seconds 10
aiget-check-env --allow-attach-reset --steps 100
aiget-random-rollout --seconds 60
aiget-debug-reward-signal --seconds 120 --send-actions
aiget-test-reset --resets 5
aiget-ptrace-il2cpp
```

---

## Design Principles

- Prefer direct, low-latency observation over slow external APIs.
- Make invalid signals explicit instead of hiding them.
- Keep reset, reward, observation, and learning separate.
- Benchmark the environment before training.
- Do not trust reward until it has passed a live preflight.
- Do not start serious learning until reset is deterministic.

---

## Roadmap

Environment layer:

- [x] Build Linux mouse action sender
- [x] Create Gymnasium-compatible environment
- [x] Add image observation
- [x] Add raw-memory cursor observation
- [x] Add reward/progress diagnostics
- [x] Add random rollout smoke testing
- [x] Add deterministic reset path for current setup
- [ ] Improve rich-state memory layout resolution
- [ ] Add shared-memory observation blob from inside the game
- [ ] Improve reset portability
- [ ] Add cleaner benchmark reports

Agent/training layer:

- [ ] Move long-running training experiments outside this repo
- [ ] Track model checkpoints separately
- [ ] Track training curves separately
- [ ] Add policy evaluation videos separately
- [ ] Compare reward shaping variants separately

---

## Important Note

This project does not include the game files.

You need a native Linux installation of *Getting Over It with Bennett Foddy* with:

```text
GettingOverIt.x86_64
UnityPlayer.so
GameAssembly.so
```

---

## Suggested Repo Description

```text
Native Linux Gymnasium environment/control stack for Getting Over It using Unity memory reads, /dev/uinput actions, reset automation, and reward debugging.
```

## Suggested Topics

```text
reinforcement-learning
gymnasium
linux
unity
il2cpp
game-ai
computer-vision
uinput
python
systems-programming
```

---

## License

Add a license before treating this as a reusable public project.
