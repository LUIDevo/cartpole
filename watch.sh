#!/usr/bin/env bash
set -euo pipefail

# Open the simulation in a WINDOW so you can see it.
#   ./watch.sh            -> watch the random policy (no client needed)
#   ./watch.sh 9999       -> control mode: window + TCP server on port 9999,
#                            then run a client, e.g.:
#                            python examples/nn_control_client.py --port 9999
#
# Needs a C#-capable Godot with a display. Override binary with $GODOT.

GODOT="${GODOT:-godot-mono}"
PROJECT="$(cd "$(dirname "$0")/simulation" && pwd)"
PORT="${1:-}"

dotnet build "$PROJECT/simulation.sln" -c Debug >/dev/null

if [[ -n "$PORT" ]]; then
  exec "$GODOT" --path "$PROJECT" -- --port="$PORT"
else
  exec "$GODOT" --path "$PROJECT"
fi
