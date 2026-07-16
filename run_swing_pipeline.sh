#!/usr/bin/env bash
# Swing-up specialist via trajectory optimization (see swing_opt.py).
# Stage A: 120 per-DR-draw iLQR trajectories -> TVLQR closed-loop
#          dataset -> BC into policy_uu_swing_ilqr.pt  (~3 h CPU)
# Stage B: honest per-attempt end-to-end eval (swing net + balance net
#          with the watch_math.py switch, wide starts, no resets).
# If stage B looks good (>60% per-attempt), promote with:
#   cp policy_uu_swing_ilqr.pt policy_uu_swing.pt
#
# Run detached so it survives closing the terminal:
#   nohup ./run_swing_pipeline.sh > swing_pipeline.log 2>&1 &
# (A laptop suspend pauses and resumes it; only shutdown kills it.)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}

echo "=== STAGE A: trajectory optimization + dataset + BC ==="
$PY swing_opt.py --starts 120 --dr-per-traj 12 --out policy_uu_swing_ilqr.pt

echo "=== STAGE B: end-to-end per-attempt eval ==="
$PY - <<'EOF'
import numpy as np
import torch
from math_env import MathCartPoleVec, HANDOFF_COS, HANDOFF_OMEGA, HALF_TRACK
from train import Network

def load(p):
    n = Network()
    n.load_state_dict(torch.load(p, weights_only=True))
    n.eval()
    return n

swing, bal = load("policy_uu_swing_ilqr.pt"), load("policy_uu.pt")
N = 512
env = MathCartPoleVec(N, seed=17, fixed_goal=(1.0, 1.0), balance_frac=0.0)
mode = np.zeros(N, dtype=bool)
handoff_at = np.full(N, -1.0)
up_ticks = np.zeros(N)
wall_ticks = np.zeros(N)
for t in range(1800):
    obs = torch.from_numpy(env.observe())
    with torch.no_grad():
        u = np.where(mode, bal(obs).numpy(), swing(obs).numpy())
    env.step(u)
    c1, c2 = np.cos(env.theta1), np.cos(env.theta2)
    enter = ~mode & (c1 > HANDOFF_COS) & (c2 > HANDOFF_COS) \
        & (np.abs(env.omega1) < HANDOFF_OMEGA) & (np.abs(env.omega2) < HANDOFF_OMEGA)
    mode |= enter
    handoff_at[enter & (handoff_at < 0)] = t / 60.0
    mode &= ~(mode & ((c1 < 0.7) | (c2 < 0.7)))
    up_ticks += (c1 > 0.9) & (c2 > 0.9)
    wall_ticks += np.abs(env.x) >= HALF_TRACK - 1.0

ok = handoff_at >= 0
print(f"single attempts (30s, wide starts, NO resets): {N}")
print(f"per-attempt handoff success: {ok.mean()*100:.0f}%  "
      f"median t {np.median(handoff_at[ok]) if ok.any() else float('nan'):.1f}s")
print(f"mean both-up: {(up_ticks/60).mean():.1f}s/30s   "
      f">15s up: {(up_ticks/60 > 15).mean()*100:.0f}%")
print(f"mean wall time: {(wall_ticks/60).mean():.2f}s")
EOF
