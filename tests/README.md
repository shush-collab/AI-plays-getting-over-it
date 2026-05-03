# Tests

This directory contains unit tests for `src/aiget`.

Current coverage is focused on:

- schema formatting
- CLI wrapper help/output
- fast cursor lane helpers
- live-layout persistence helpers
- v1 observation vector helpers
- slow raw rich-state batching helpers
- Gym Dict observation, benchmark, check, rollout, and training modules compile
- Gym/action modules compile and are dependency-backed through `pyproject.toml`
- ptrace helper safety utilities that can be checked without a live game process

Run the suite with:

```bash
python3 -m unittest discover -s tests -v
```

These tests should stay independent of the running game process.
