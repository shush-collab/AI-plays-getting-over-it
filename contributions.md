# Contributing

This repository is building Linux-first runtime tooling for reinforcement learning experiments on
*Getting Over It with Bennett Foddy*. Current work is focused on live state extraction,
observation building, reverse-engineering notes, and the infrastructure needed before full
rollout collection and training.

## Repository Layout

```text
AIget/
├── docs/
├── src/
│   └── aiget/
├── tests/
├── goi_live_position.py
├── goi_memory_probe.py
├── goi_observation_schema.py
├── goi_observation_state.py
├── goi_ptrace_il2cpp.py
├── pyproject.toml
├── README.md
└── contributions.md
```

## What Goes Where

- `src/aiget/` contains the real implementation.
- Root-level `goi_*.py` files are thin compatibility launchers.
- `docs/` contains design notes, reverse-engineering details, and runtime assumptions.
- `tests/` contains automated checks that should not require a live game process.
- `docs/observation-roadmap.md` tracks the current observation-architecture work.

## Working Agreement

- Keep implementation code inside `src/aiget/`.
- Keep root-level launcher scripts minimal.
- Prefer small, single-purpose functions.
- Add tests for non-trivial changes when practical.
- Preserve Linux-first assumptions unless a change intentionally broadens support.
- Avoid hidden side effects during import.
- Keep CLI parsing near `main()`.
- Keep reusable logic out of wrappers and inside package code.

## Code Style

- Target Python 3.12+.
- Prefer the standard library first.
- Use clear names for modules, functions, and variables.
- Add type hints for function parameters and return values.
- Prefer explicit inputs and outputs over global state.
- Keep comments rare and useful.
- Explain why when the code is not self-evident.
- Avoid duplicated logic when a shared helper would make the code clearer.

## Reverse-Engineering And Runtime-State Changes

This repository includes work that depends on reverse-engineered runtime details.

If your change affects memory paths, offsets, live object resolution, coordinate extraction, or
observation features:

- Clearly separate confirmed results from guesses or inferences.
- Do not present unverified assumptions as facts.
- Document the validation method used.
- Update the relevant file in `docs/` if assumptions or validated paths changed.
- Keep the fast cursor lane and the slow rich-state lane conceptually separate unless your change is intentionally unifying them.

Good contribution notes include:

- What was changed.
- What was validated manually.
- Under what conditions it was validated.
- What remains uncertain.

## Adding A New Module

When adding a new module:

- Put reusable code under `src/aiget/`.
- Name the file after its responsibility.
- Add a short module docstring if needed.
- Keep CLI parsing in `main()`.
- Add a console script entry in `pyproject.toml` only when the module should be user-facing.

Examples:

- Good: `src/aiget/terrain_sampling.py`
- Avoid: `src/aiget/new_code.py`

## Adding New Functions

When adding new functions:

- Choose a name that describes the result, not the implementation detail.
- Add type hints.
- Raise clear exceptions for invalid states.
- Keep the function focused.
- Split functions that try to do parsing, validation, I/O, and formatting all at once.

Examples:

- Good: `def build_observation_frame(state: GameState) -> list[float]:`
- Avoid: `def do_everything(x):`

## CLI Conventions

Every user-facing CLI module should:

- Expose a `main()` function.
- Provide actionable error messages.
- Keep human-readable and machine-readable output modes explicit.
- Flush output consistently for long-running streams.
- Continue to work with `--help`.
- Make playability-impacting polling or refresh rates explicit in CLI flags rather than hidden constants.

## Tests

Before submitting a change, run:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall src *.py
```

If you have dev tools installed, also run:

```bash
pytest
ruff check .
```

Test expectations:

- Add or update tests for non-trivial behavior that can be checked without the live game process.
- Keep tests focused and stable.
- Avoid tests that pretend to verify live gameplay or memory-reading when they do not actually do
  so.

## Documentation

Update documentation when needed:

- Update `README.md` if setup, usage, or project structure changed.
- Update `docs/` when behavior depends on reverse-engineering details or runtime offsets.
- Remove stale instructions or references when the architecture changes instead of leaving both old and new guidance in place.
- Keep documentation concrete and close to the code or behavior it explains.

## Pull Request Checklist

Before opening a PR, check that:

- The new code lives in the right layer.
- Public behavior is documented.
- Non-trivial logic has coverage where practical.
- CLI behavior still works.
- The change does not include unrelated cleanup or formatting churn.
- Reverse-engineering changes are clearly marked as confirmed, inferred, or unverified where
  relevant.

## Preferred Contribution Style

The best contributions to this project are:

- Small.
- Well-scoped.
- Easy to review.
- Easy to test.
- Explicit about what is known and what is still uncertain.
