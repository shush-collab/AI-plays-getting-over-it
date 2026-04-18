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
├── goi_ptrace_il2cpp.py
├── .gitignore
├── pyproject.toml
└── README.md
```

The root-level `goi_*.py` files are thin compatibility wrappers. The actual implementation now lives in `src/aiget/`.

## Current Stage

The immediate blocker was solved: we can now read the live in-game `x,y` position of the player controller in real time from process memory.

What is working now:
- `src/aiget/ptrace_il2cpp.py` resolves the live `PlayerControl` object, its `fakeCursorRB` field, and queries Unity for the true `Rigidbody2D.position`.
- `src/aiget/live_position.py` uses that ground truth once at startup, calibrates the matching raw-memory path, then streams `x,y` from `/proc/<pid>/mem`.
- `src/aiget/memory_probe.py` is a helper for memory inspection and direct vector watching.
- `src/aiget/observation_schema.py` defines the planned RL observation vector layout.

The current validated raw-memory path is:
- `fakeCursorRB_native + 0xA8`

## Quick Start

Use the project venv:

```bash
source ~/Documents/AIget/venv/bin/activate
```

Optional editable install for package-style imports:

```bash
pip install -e .
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
