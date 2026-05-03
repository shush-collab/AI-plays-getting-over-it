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
│       ├── live_position.py
│       ├── memory_probe.py
│       ├── observation_schema.py
│       ├── observation_state.py
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

The current focus is live observation architecture for RL.

What is working now:
- `src/aiget/ptrace_il2cpp.py` resolves the live `PlayerControl` object, its `fakeCursorRB` field, and queries Unity for the true `Rigidbody2D.position`.
- `src/aiget/live_position.py` freezes the validated raw-memory cursor path into an explicit fast observation lane and streams `x,y` from `/proc/<pid>/mem`.
- `src/aiget/live_layout.py` freezes a reusable rich raw-memory layout at startup and can save/load/validate it.
- `src/aiget/memory_probe.py` is a helper for memory inspection and direct vector watching.
- `src/aiget/observation_schema.py` defines the planned RL observation vector layout.
- `src/aiget/observation_state.py` uses startup discovery plus two raw-memory live lanes:
  - a fast cursor lane from raw memory
  - a slower raw rich-state lane for body, hammer, contacts, and progress
- The live rich lane is packaged into one `RichStateSnapshot` object and updated in the background through persistent `MemReader` instances only.
- When discovery misses a moving calibration sample, the live stream falls back to the currently validated cursor path `fakeCursorRB_native + 0xA8`.
- Optional startup validation can compare the resolved raw layout to authoritative ptrace samples once and then exit the expensive path.
- `goi_observation_state.py` is now cache-first for rich layout startup and will fail fast to a partial layout instead of blocking on optional rich-field discovery.
- The fast loop now reuses the latest rich snapshot without blocking on fresh background updates.
- The current observation architecture roadmap lives in `docs/observation-roadmap.md`.

The current validated raw-memory path is:
- `fakeCursorRB_native + 0xA8`

What is still not solved:

- Some rich fields still cannot be resolved to raw memory and are emitted as zero/default values with `rich_state_valid_mask` set to `false`.
- The default startup path only resolves must-have rich fields; optional fields such as body angle and hammer anchor/tip stay invalid unless explicitly requested.
- Higher rich-snapshot refresh rates can still hurt playability even though the live path is now raw-memory only.
- The final RL-facing observation builder API does not exist yet.
- The eventual in-game exported observation blob does not exist yet.

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

- [ ] Finish the remaining raw-readable observation fields, especially body-contact state and the synthetic LIDAR features.
- [ ] Implement synthetic LIDAR / raycasting against the game world.
- [ ] Wire control inputs into the game process for training.
- [ ] Build the unified observation + action loop for rollout collection.
- [ ] Start data collection and training.
