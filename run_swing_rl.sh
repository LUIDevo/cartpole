#!/usr/bin/env bash
# Swing-up via energy-pump warm-start + PPO --handoff fine-tune.
#
# Replaces the iLQR->BC pipeline (run_swing_pipeline.sh), which failed:
# 120 independent TVLQR trajectories are multimodal, so BC of them floors
# at mse ~0.39 and yields 8% honest eval. And pure analytic energy+LQR
# caps ~19% sustained hold because energy shaping cannot steer both links
# into a slow both-upright arrival (see swing-analytic-ceiling finding).
#
# Stage A: BC the single-mode analytic energy controller into the actor
#          (~1-2 min) -> policy_uu_swing_bc.pt. This is a clean prior:
#          it already delivers the poles to the top region.
# Stage B: PPO --handoff fine-tunes from that prior. HANDOFF_BONUS rewards
#          delivering a catchable state; PPO learns the near-top
#          configuration control the energy law lacks.
#
# Detached run that survives closing the terminal:
#   nohup ./run_swing_rl.sh > swing_rl.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}

echo "=== STAGE A: energy-pump BC warm-start ==="
$PY swing_bc.py --out policy_uu_swing_bc.pt

echo "=== STAGE B: PPO --handoff fine-tune ==="
$PY train.py --goal uu --handoff \
    --init policy_uu_swing_bc.pt \
    --out policy_uu_swing.pt \
    --log training_log_uu_swing.csv \
    --iters 400 --balance-frac 0 --max-steps 3000

echo "=== eval: honest per-attempt swing+balance ==="
$PY - <<'EOF'
import numpy as np, torch
from math_env import MathCartPoleVec, HANDOFF_COS, HANDOFF_OMEGA
from train import Network
def load(p):
    n=Network(); n.load_state_dict(torch.load(p,weights_only=True)); n.eval(); return n
swing, bal = load("policy_uu_swing.pt"), load("policy_uu.pt")
RELEASE_COS = 0.7
N=512; env=MathCartPoleVec(N,seed=17,fixed_goal=(1.0,1.0),balance_frac=0.0)
mode=np.zeros(N,bool); hoff=np.full(N,-1.0); up_tail=np.zeros(N)
for t in range(1800):
    with torch.no_grad():
        ob=torch.from_numpy(env.observe())
        u=np.where(mode, bal(ob).numpy(), swing(ob).numpy())
    env.step(u)
    c1,c2=np.cos(env.theta1),np.cos(env.theta2)
    enter=~mode&(c1>HANDOFF_COS)&(c2>HANDOFF_COS)&(np.abs(env.omega1)<HANDOFF_OMEGA)&(np.abs(env.omega2)<HANDOFF_OMEGA)
    mode|=enter; hoff[enter&(hoff<0)]=t/60
    mode&=~(mode&((c1<RELEASE_COS)|(c2<RELEASE_COS)))
    if t>=900: up_tail+=(c1>0.9)&(c2>0.9)
c=hoff>=0
print(f"per-attempt handoff: {100*c.mean():.0f}%   held-last-15s: {100*(up_tail/900>0.9).mean():.0f}%")
EOF
