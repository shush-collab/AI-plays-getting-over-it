#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs/reset_test

resets="${1:-1}"
log="${LOG:-runs/reset_test/reset_probe_$(date +%Y%m%d_%H%M%S).log}"
startup_key="${STARTUP_KEY:-}"
window_left="${WINDOW_LEFT:-320}"
window_top="${WINDOW_TOP:-178}"
window_width="${WINDOW_WIDTH:-1920}"
window_height="${WINDOW_HEIGHT:-1080}"

startup_key_args=()
if [[ -n "$startup_key" ]]; then
  startup_key_args=(--startup-key "$startup_key")
fi

./venv/bin/python -m aiget.test_reset \
  --resets "$resets" \
  --clean-save-path "${CLEAN_SAVE_PATH:-$HOME/goi_reset_saves/start_clean}" \
  --active-save-path "${ACTIVE_SAVE_PATH:-$HOME/.config/unity3d/Bennett Foddy/Getting Over It}" \
  --capture-left "${CAPTURE_LEFT:-$window_left}" \
  --capture-top "${CAPTURE_TOP:-$window_top}" \
  --capture-width "${CAPTURE_WIDTH:-$window_width}" \
  --capture-height "${CAPTURE_HEIGHT:-$window_height}" \
  --window-left "$window_left" \
  --window-top "$window_top" \
  --window-width "$window_width" \
  --window-height "$window_height" \
  --startup-mode auto \
  --startup-delay "${STARTUP_DELAY:-2}" \
  --title-click "${TITLE_CLICK_X:-1290}" "${TITLE_CLICK_Y:-305}" \
  --confirm-click "${CONFIRM_CLICK_X:-640}" "${CONFIRM_CLICK_Y:-430}" \
  "${startup_key_args[@]}" \
  --startup-attempts "${STARTUP_ATTEMPTS:-2}" \
  --game-ready-timeout "${GAME_READY_TIMEOUT:-90}" \
  --sleep-after-reset "${SLEEP_AFTER_RESET:-2}" \
  --save-reset-trace \
  2>&1 | tee "$log" | awk -v log_path="$log" '
    /^RESET [0-9]+/ ||
    /FAILED/ ||
    /failure_stage:/ ||
    /reason:/ ||
    /pid:/ ||
    /reset_mode:/ ||
    /startup_action_sent:/ ||
    /startup_state:/ ||
    /playercontrol_ready_ms:/ ||
    /fast_cursor_addr:/ ||
    /image_mean:/ ||
    /image_std:/ ||
    /process_lost:/ ||
    /frame:/ ||
    /playable_frame:/ { print }
    END { print "log: " log_path }
  '
