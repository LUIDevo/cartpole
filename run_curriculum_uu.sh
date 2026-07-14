#!/usr/bin/env bash
# Curriculum training for the up-up specialist.
#
# Stage 0 distills an LQR balance teacher into the network (~3 min):
# PPO cannot discover the up-up balance by exploration, but it preserves
# and extends the skill once cloned in.
# Stage 1 fine-tunes on balance-heavy short episodes so the catch
# generalizes beyond the teacher (~30-60 min).
# Stage 2 trains the full task: swing-up connects to the cloned catch.
# std-init 0.15 there is a compromise: enough exploration to learn the
# swing (energy shaping supplies the gradient), low enough not to wreck
# the balance. If swing-up doesn't appear, raise it; if the hold decays,
# lower it.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}

$PY distill.py --out policy_uu_bc.pt

$PY train.py --goal uu --init policy_uu_bc.pt --out policy_uu_stage1.pt \
    --log training_log_uu_stage1.csv \
    --iters 300 --balance-frac 0.5 --max-steps 600 \
    --std-init 0.08 --std-min 0.05 --lr 1e-4 --critic-warmup 5

$PY train.py --goal uu --init policy_uu_stage1.pt --out policy_uu.pt \
    --log training_log_uu.csv \
    --iters 1500 --balance-frac 0.25 --max-steps 3000 \
    --std-init 0.15 --std-min 0.05 --lr 1e-4 --critic-warmup 2
