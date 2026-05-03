# AIget

This project is building a reinforcement learning agent for *Getting Over It with Bennett Foddy* on native Linux.

The installed game is a native Linux Unity build with:

- `GettingOverIt.x86_64`
- `UnityPlayer.so`
- `GameAssembly.so`

## Repository Layout

```text
AIget/
├── docs/
│   ├── observation-schema.md
│   ├── observation-roadmap.md
│   └── player-movement.md
├── src/
│   └── aiget/
│       ├── __init__.py
│       ├── action_sender.py
│       ├── benchmark_observation.py
│       ├── check_env.py
│       ├── env.py
│       ├── frame_capture.py
│       ├── live_layout.py
│       ├── live_position.py
│       ├── memory_probe.py
│       ├── observation_vector.py
│       ├── observation_schema.py
│       ├── observation_state.py
│       ├── ptrace_il2cpp.py
│       ├── random_rollout.py
│       └── train_sac.py
├── tests/
│   └── README.md
├── goi_live_position.py
├── goi_memory_probe.py
├── goi_observation_schema.py
├── goi_observation_state.py
├── goi_ptrace_il2cpp.py
├── .gitignore
├── pyproject.toml
└── README.md
```

The root-level `goi_*.py` files are thin compatibility wrappers. The actual implementation now lives in `src/aiget/`.

Development and contribution guidance lives in `contributions.md`.

## Current Stage

The current focus is a usable Gymnasium environment for RL.

What is working now:
- `src/aiget/env.py` exposes `GettingOverItEnv`, a Gymnasium environment with Dict observations and normalized 2D mouse actions.
- `src/aiget/action_sender.py` sends relative mouse movement through Linux `/dev/uinput`.
- `src/aiget/observation_vector.py` builds the v1 fixed 32-float state vector.
- `src/aiget/frame_capture.py` provides an 84x84 grayscale `uint8` image observation.
- `src/aiget/ptrace_il2cpp.py` resolves the live `PlayerControl` object, its `fakeCursorRB` field, and queries Unity for the true `Rigidbody2D.position`.
- `src/aiget/live_position.py` freezes the validated raw-memory cursor path into an explicit fast observation lane.
- `src/aiget/live_layout.py` freezes a reusable rich raw-memory layout at startup and can save/load/validate it.
- `src/aiget/memory_probe.py` is a helper for memory inspection and direct vector watching.
- `src/aiget/observation_schema.py` defines the v1 RL observation vector layout.
- `src/aiget/observation_state.py` uses startup discovery plus two raw-memory live lanes:
  - a fast cursor lane from raw memory
  - a slower raw rich-state lane for body, hammer tip/direction, and progress
- `src/aiget/benchmark_observation.py` reports fast/rich rates plus active work, wall time, sleep time, and missed deadlines.
- `src/aiget/check_env.py` runs SB3 `check_env` plus a short random-action smoke validation.
- `src/aiget/random_rollout.py` runs random-action smoke rollouts and writes CSV metrics.
- `src/aiget/train_sac.py` is a guarded SB3 `MultiInputPolicy` training entrypoint.
- The training hot path uses fixed-address raw reads, preallocated numpy arrays, and no JSON serialization.
- The image lane and rich lane update in background threads; `env.step()` consumes the latest snapshots without waiting for fresh ones.
- `action_repeat` defaults to `2`, so the default 30Hz frame loop gives about 15 policy decisions per second.
- When discovery misses a moving calibration sample, the live stream falls back to the currently validated cursor path `fakeCursorRB_native + 0xA8`.
- Optional startup validation can compare the resolved raw layout to authoritative ptrace samples once and then exit the expensive path.
- Runtime play does not call ptrace, Unity, or IL2CPP functions.
- The environment reuses the latest rich snapshot without blocking on fresh background updates.
- The current observation architecture roadmap lives in `docs/observation-roadmap.md`.

The current validated raw-memory path is:
- `fakeCursorRB_native + 0xA8`

What is still not solved:

- Some rich fields still cannot be resolved to raw memory and are emitted as zero/default values with `rich_state_valid_mask` set to `false`.
- The current default reset is attach-only. It clears Python episode state but does not restore the game to a known checkpoint.
- The eventual in-game exported observation blob does not exist yet.
- Real training should not run until a reliable game reset/checkpoint restore exists.

## Quick Start

This project declares its Python dependencies in `pyproject.toml`.

- `pyproject.toml` is the source of truth for declared dependencies and project metadata.
- `uv.lock` pins the fully resolved dependency set for reproducible installs with `uv`.
- Tools such as `uv` or `pip` read `pyproject.toml`; `uv` can additionally use `uv.lock`.

## Setup With `uv`

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment:

```bash
uv venv
```

Activate it:

```bash
source .venv/bin/activate
```

Sync the default environment from the lockfile:

```bash
uv sync
```

Sync including developer dependencies:

```bash
uv sync --extra dev
```

Sync including RL dependencies for SB3 validation/training:

```bash
uv sync --extra rl
```

If you prefer a single command without manual activation, you can also run commands through
`uv` directly:

```bash
uv run goi_observation_schema.py --format markdown
uv run --extra dev python -m unittest discover -s tests -v
uv run --extra rl python -m aiget.check_env --allow-attach-reset --steps 100
```

If dependency declarations change, regenerate the lockfile:

```bash
uv lock
```

## Setup With `pip`

If you want to use `pip` instead of `uv`, `pyproject.toml` remains the source of truth, but
`pip` does not use `uv.lock`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"
pip install -e ".[rl]"
```

## Usage

Gymnasium environment:

```python
from aiget.env import GettingOverItEnv

env = GettingOverItEnv(dt=1.0 / 30.0)
obs, info = env.reset()
state = obs["state"]  # float32, shape=(32,)
image = obs["image"]  # uint8, shape=(84, 84, 1)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

Observation benchmark:

```bash
python -m aiget.benchmark_observation --seconds 10
```

SB3 environment smoke validation:

```bash
python -m aiget.check_env --allow-attach-reset --steps 100
```

Random rollout smoke test:

```bash
python -m aiget.random_rollout --seconds 60 --csv runs/random_rollout.csv
```

Guarded SAC/PPO training entrypoint:

```bash
python -m aiget.train_sac --algo sac --steps 10000
```

Training refuses by default because real game reset is not implemented. Use
`--allow-attach-reset` only for smoke tests, not real learning runs.

Fast cursor lane only:

```bash
python goi_live_position.py --format json
```

Fast cursor lane, emit only on change:

```bash
python goi_live_position.py --format json --only-changes
```

Fast cursor lane, follow restarts and recalibrate automatically:

```bash
python goi_live_position.py --format json --follow-restarts
```

Unity ground-truth resolver:

```bash
python goi_ptrace_il2cpp.py
```

Memory probe helper:

```bash
python goi_memory_probe.py icalls
```

Observation schema:

```bash
python goi_observation_schema.py --format markdown
```

One-shot rich observation payload:

```bash
python goi_observation_state.py --format json
```

Continuous split-lane observation stream:

```bash
python goi_observation_state.py --format text --samples 0 --interval 0.05 --rich-snapshot-interval 0.2
```

Resolve once, validate once, and save the raw rich layout for the current PID:

```bash
python goi_observation_state.py --format json --validate-layout --live-layout-cache /tmp/goi-layout.json
```

Fast startup with the default cache-first partial-layout policy:

```bash
python goi_observation_state.py --format json --layout-discovery-timeout 1.5
```

Echo a previous action into the payload:

```bash
python goi_observation_state.py --format json --previous-action 0.0 0.0
```

Installed console scripts are available after `pip install -e .`:

```bash
aiget-live-position --format json
aiget-memory-probe icalls
aiget-observation-schema --format markdown
aiget-observation-state --format json
aiget-benchmark-observation --seconds 10
aiget-check-env --allow-attach-reset --steps 100
aiget-random-rollout --seconds 60
aiget-train-sac --algo sac --steps 10000
aiget-ptrace-il2cpp
```

## Verification

Repository-level checks that do not require the game process:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall src *.py
```

## Todo

Observation architecture progress is tracked in `docs/observation-roadmap.md`.

Remaining project backlog:

- [x] Wire v1 control inputs through Linux uinput.
- [x] Build the first unified observation + action environment API.
- [x] Add image observation, action repeat, benchmark timing, and smoke rollout tools.
- [ ] Implement reliable game reset/checkpoint restore.
- [ ] Start real training only after reset works.
- [ ] Replace external rich reads with an in-game BepInEx IL2CPP plugin writing one shared-memory observation blob.
