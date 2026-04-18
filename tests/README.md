# Tests

This directory contains unit tests for `src/aiget`.

Current coverage is focused on:

- schema formatting
- CLI wrapper help/output
- fast cursor lane helpers
- slow rich-state batching helpers
- ptrace helper safety utilities that can be checked without a live game process

Run the suite with:

```bash
python3 -m unittest discover -s tests -v
```

These tests should stay independent of the running game process.
