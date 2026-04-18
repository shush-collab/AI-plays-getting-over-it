# AIget

This project is building a reinforcement learning agent for *Getting Over It with Bennett Foddy* on native Linux.

## Repository Layout

```text
AIget/
├── docs/
│   ├── observation-schema.md
│   └── player-movement.md
├── src/
│   └── aiget/
│       ├── __init__.py
│       ├── live_position.py
│       ├── memory_probe.py
│       ├── observation_schema.py
│       └── ptrace_il2cpp.py
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

The immediate blocker was solved: we can now read the live in-game `x,y` position of the player controller in real time from process memory.

What is working now:
- `src/aiget/ptrace_il2cpp.py` resolves the live `PlayerControl` object, its `fakeCursorRB` field, and queries Unity for the true `Rigidbody2D.position`.
- `src/aiget/live_position.py` uses that ground truth once at startup, calibrates the matching raw-memory path, then streams `x,y` from `/proc/<pid>/mem`.
- `src/aiget/memory_probe.py` is a helper for memory inspection and direct vector watching.
- `src/aiget/observation_schema.py` defines the planned RL observation vector layout.
- `src/aiget/observation_state.py` streams the currently implemented observation features: cursor position, cursor velocity, and progress features.
- When discovery misses a moving calibration sample, the live stream falls back to the currently validated cursor path `fakeCursorRB_native + 0xA8`.

The current validated raw-memory path is:
- `fakeCursorRB_native + 0xA8`

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

If you prefer a single command without manual activation, you can also run commands through
`uv` directly:

```bash
uv run goi_observation_schema.py --format markdown
uv run --extra dev python -m unittest discover -s tests -v
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
```

## Usage

Live JSON stream for RL:

```bash
python goi_live_position.py --format json
```

Only emit on change:

```bash
python goi_live_position.py --format json --only-changes
```

Follow restarts and recalibrate automatically:

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

Implemented observation-state stream:

```bash
python goi_observation_state.py --format json
```

Installed console scripts are also available after `uv sync` or `pip install -e .`:

```bash
aiget-live-position --format json
aiget-memory-probe icalls
aiget-observation-schema --format markdown
aiget-observation-state --format json
aiget-ptrace-il2cpp
```

## Verification

Repository-level checks that do not require the game process:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall src *.py
```

## Todo

- [x] Resolve the live `PlayerControl` object and confirm the correct movement source.
- [x] Resolve `fakeCursorRB` and verify the live `Rigidbody2D.position`.
- [x] Calibrate a direct raw-memory read path and stream `x,y` from `/proc/<pid>/mem`.
- [x] Define the RL observation schema. The current `v1` schema is documented in `src/aiget/observation_schema.py` and covers body state, hammer state, cursor state, contact state, progress, previous action, and synthetic LIDAR distances.
- [ ] Add the extra state readers required by that observation schema.
- [ ] Implement synthetic LIDAR / raycasting against the game world.
- [ ] Wire control inputs into the game process for training.
- [ ] Build the observation + action loop for rollout collection.
- [ ] Start data collection and training.
