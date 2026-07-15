#!/usr/bin/env bash
# Curriculum training for the up-up specialist.
#
# Stage 0 distills an LQR balance teacher into the network (~3 min):
# PPO cannot discover the up-up balance by exploration, but it preserves
# and extends the skill once cloned in.
# Stage 1 fine-tunes on balance-heavy short episodes so the catch
# generalizes beyond the teacher (~30-60 min).
# Stage 2 trains a separate swing specialist from scratch: episodes end
# with a bonus the moment both poles enter the balance net's catch basin
# (--handoff), so delivering a catchable state is its whole objective.
# At runtime watch_math.py switches between the two nets automatically.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}

$PY distill.py --out policy_uu_bc.pt

$PY train.py --goal uu --init policy_uu_bc.pt --out policy_uu.pt \
    --log training_log_uu_balance.csv \
    --iters 300 --balance-frac 0.5 --max-steps 600 \
    --std-init 0.08 --std-min 0.05 --lr 1e-4 --critic-warmup 5

$PY train.py --goal uu --out policy_uu_swing.pt \
    --log training_log_uu_swing.csv \
    --iters 1000 --balance-frac 0 --max-steps 3000 --handoff
