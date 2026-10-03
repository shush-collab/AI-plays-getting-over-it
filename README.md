# AIget

Small Linux tooling for an owned *Getting Over It* installation. It sends
mouse input, reads live Unity/IL2CPP state, and verifies relaunch/save-reset
readiness. It is not a simulator or an RL training project.

## Test the tools

```sh
uv sync --extra dev
uv run pytest -q
```

To test a real relaunch, provide an owned game installation, a known-clean save
copy, the active Unity save location, and the exact capture rectangle of the
game window:

```sh
uv run aiget-test-reset \
  --resets 1 \
  --launch-command steam -applaunch 240720 -- -screen-fullscreen 0 -screen-width 1280 -screen-height 720 \
  --clean-save-path "$HOME/goi_reset_saves/start_clean" \
  --active-save-path "$HOME/.config/unity3d/Bennett Foddy/Getting Over It" \
  --capture-left 320 --capture-top 178 --capture-width 1280 --capture-height 720 \
  --save-reset-trace
```

On X11/Xwayland, install `xdotool` and `xwd`. On Wayland, install and run
`ydotool`; the reset command selects it automatically when available. Output
frames and traces are written under `runs/reset_test/`.

Useful state probes:

```sh
uv run aiget-live-position --help
uv run aiget-memory-probe --help
uv run aiget-observation-state --help
uv run aiget-ptrace-il2cpp --help
```
