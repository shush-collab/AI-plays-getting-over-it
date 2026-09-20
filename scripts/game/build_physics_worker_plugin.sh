#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
game_root="${1:-$repo_root/runs/training-game/getting-over-it-physics-worker}"
project="$repo_root/game/worker-plugin/AIget.PhysicsWorker/AIget.PhysicsWorker.csproj"
output="$repo_root/game/worker-plugin/AIget.PhysicsWorker/bin/Release/netstandard2.1/AIget.PhysicsWorker.dll"
plugin_directory="$game_root/BepInEx/plugins"
dotnet_command="${DOTNET_COMMAND:-dotnet}"

"$dotnet_command" build "$project" --configuration Release --property:GameRoot="$game_root"
install -D -m 0644 "$output" "$plugin_directory/AIget.PhysicsWorker.dll"
