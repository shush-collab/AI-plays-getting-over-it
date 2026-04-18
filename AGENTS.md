# AGENTS.md

## Purpose

This repository contains Linux-first runtime tooling for reinforcement learning experiments on
*Getting Over It with Bennett Foddy*.

The project is currently in the instrumentation and observation-building stage. It is not yet a
complete training pipeline. Current work is focused on:

1. reading live game state reliably
2. expanding the observation schema with additional state readers
3. adding synthetic LIDAR / raycasting
4. wiring control inputs into the game process
5. building the observation + action rollout loop
6. starting data collection and training

## Read this first

Before making any change:

1. Read `README.md`
2. Read `contributions.md`
3. Read any relevant file in `docs/`
4. Identify the smallest possible useful change
5. Prefer explanation and verification over broad edits

## Repository structure

- Real implementation code belongs in `src/aiget/`
- Root-level `goi_*.py` files are thin compatibility wrappers only
- `docs/` contains reverse-engineering notes and schema documentation
- `tests/` contains checks that should not require a live game process

## Project assumptions

- Python 3.12+
- Linux-first behavior is intentional
- Import-time side effects should be avoided
- User-facing CLI modules should expose a `main()` function
- Standard library solutions are preferred unless a new dependency is clearly justified

## Rules for code changes

- Keep changes small and focused
- Put implementation code in `src/aiget/`
- Keep root wrappers minimal
- Do not refactor unrelated code
- Do not rename files or move modules unless necessary
- Do not add dependencies unless they clearly simplify necessary work
- Prefer small, explicit functions with type hints
- Runtime work should start from `main()`, not from import side effects

## Rules for reverse-engineering and runtime-state work

For any change involving memory paths, offsets, runtime objects, coordinate extraction, or
observation features:

- Clearly distinguish between `confirmed`, `inferred`, and `unverified`.
- Do not present guesses as facts
- Change one thing at a time
- Preserve existing validated behavior unless intentionally replacing it
- If behavior or assumptions change, update the relevant file in `docs/`

## Validation expectations

Run the safest relevant checks after making changes.

Minimum repository-level checks:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall src *.py
```

If available, also run:

```bash
pytest
ruff check .
```

## Environment setup

Preferred setup with `uv`:

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

Alternative setup with `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Documentation expectations

Update documentation when behavior changes:

- Update `README.md` if setup, usage, or repo layout changes
- Update files in `docs/` when reverse-engineering details, offsets, or schema assumptions change
- Keep documentation concrete and evidence-based

## Testing expectations

For non-trivial behavior that can be checked without a live game process:

- Add or update automated tests where practical
- Keep tests focused and stable
- Avoid fake tests that pretend to verify live runtime behavior when they do not

For live game-memory changes:

- Include a short note describing what was validated manually
- State under what conditions it was validated
- State what remains uncertain

## Output format for completed work

After making a change, always report:

- What changed
- Why that change was made
- What commands were run
- What was actually verified
- What remains uncertain or unverified

## What to avoid

- Large unrelated refactors
- Sweeping formatting-only churn
- Adding abstractions before they are needed
- Claiming performance or accuracy improvements without evidence
- Claiming a reverse-engineering result is correct without validation
