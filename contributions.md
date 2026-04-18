# Contributing

This repository is structured as a Python package with thin root-level launcher scripts.
Production code belongs in `src/aiget/`. The top-level `goi_*.py` files should stay as
small entrypoint wrappers only.

## Working Agreement

- Keep implementation code inside `src/aiget/`.
- Keep root-level scripts minimal. They should import and call package `main()` functions.
- Prefer small, single-purpose functions with explicit inputs and outputs.
- Add or update tests for every non-trivial behavior change that can be verified without the
  game process.
- Preserve the existing Linux-first assumptions unless a change intentionally broadens support.
- Avoid hidden side effects during import. Runtime work should start from `main()`.

## Repository Layout

- `src/aiget/`: real application code
- `tests/`: automated checks that do not require a live game process
- `docs/`: implementation notes and design references
- `goi_*.py`: compatibility launchers for local execution from the repository root

## Adding New Files

When adding a new module:

1. Put reusable code under `src/aiget/`.
2. Name the file after its responsibility, not after an ad hoc task.
3. Add a short module docstring if the purpose is not obvious from the name.
4. Keep CLI parsing in `main()` and keep reusable logic in regular functions or classes.
5. If the module needs to be executable by users, add a console script entry in
   `pyproject.toml` and only add a root wrapper if there is a compatibility reason.

Examples:

- Good: `src/aiget/terrain_sampling.py`
- Avoid: `src/aiget/new_code.py`

## Adding New Functions

- Start with a function name that describes the result, not the implementation detail.
- Add type hints to parameters and return values.
- Raise clear exceptions for invalid states instead of returning ambiguous sentinel values.
- Keep functions focused. If a function starts doing parsing, validation, I/O, and formatting,
  split it.
- Prefer passing dependencies in as arguments where practical instead of reaching into globals.

Examples:

- Good: `def build_observation_frame(state: GameState) -> list[float]:`
- Avoid: `def do_everything(x):`

## Code Style

- Target Python 3.12+.
- Use standard library features first.
- Keep imports grouped and stable.
- Prefer dataclasses for structured records that mostly hold data.
- Prefer immutable data where it keeps the code easier to reason about.
- Keep comments rare and useful. Explain why when the code is not self-evident.
- Avoid large blocks of duplicated logic. Extract shared helpers instead.

## CLI Conventions

- Every user-facing CLI module should expose a `main()` function.
- Argument parsing should stay close to `main()`.
- Error messages should be actionable and precise.
- Long-running streams should flush output consistently.
- Human-readable and machine-readable output modes should stay explicit.

## Testing Expectations

Run these before submitting changes:

```bash
python -m unittest discover -s tests -v
python -m compileall src *.py
```

If `pytest` and `ruff` are installed, also run:

```bash
pytest
ruff check .
```

For game-memory changes, include a short note describing what was validated manually and under
what conditions.

## Documentation Expectations

- Update `README.md` when setup, usage, or repository layout changes.
- Update `docs/` when behavior depends on reverse-engineering details or runtime offsets.
- Document assumptions and limitations near the code that depends on them.

## Review Checklist

Before considering a change done, check:

- The new code lives in the right layer.
- Public behavior is documented.
- Non-trivial logic has automated coverage where feasible.
- CLI behavior still works with `--help`.
- The change does not introduce unrelated formatting churn.
