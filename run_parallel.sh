#!/usr/bin/env bash
set -euo pipefail

# Parallel headless cart-pole dataset generation.
# One Godot process per shard (true multicore); each writes its own CSV, then merged.
#
# Usage:   ./run_parallel.sh [num_shards] [episodes_per_shard] [ticks_per_episode] [out_dir]
# Example: ./run_parallel.sh 8 200 500 data
# Godot binary: set $GODOT if "godot" isn't on PATH (e.g. GODOT=/path/to/Godot_v4.6).

SHARDS="${1:-$(nproc)}"
EPISODES="${2:-100}"
TICKS="${3:-500}"
OUTDIR="${4:-data}"

# Must be a C#-capable Godot build (e.g. godot-mono). Plain "godot" can't load scripts.
GODOT="${GODOT:-godot-mono}"
PROJECT="$(cd "$(dirname "$0")/simulation" && pwd)"

mkdir -p "$OUTDIR"

echo "Building C# solution..."
dotnet build "$PROJECT/simulation.sln" -c Debug >/dev/null

echo "Launching $SHARDS shards x $EPISODES episodes x $TICKS ticks (out: $OUTDIR)..."
pids=()
for ((k=0; k<SHARDS; k++)); do
  "$GODOT" --headless --path "$PROJECT" -- \
    --out="$OUTDIR/shard_$k.csv" --episodes="$EPISODES" --ticks="$TICKS" --seed="$k" &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "Merging shards -> $OUTDIR/dataset.csv"
merged="$OUTDIR/dataset.csv"
first=1
: > "$merged"
for ((k=0; k<SHARDS; k++)); do
  f="$OUTDIR/shard_$k.csv"
  [[ -f "$f" ]] || continue
  if [[ $first -eq 1 ]]; then cat "$f" >> "$merged"; first=0
  else tail -n +2 "$f" >> "$merged"; fi   # skip repeated header
done

echo "Done. Total rows: $(( $(wc -l < "$merged") - 1 ))"
