#!/usr/bin/env python3
"""Train a catcher on the swing controller's ACTUAL arrival distribution.

The balance net policy_uu.pt was distilled from an LQR teacher on a narrow
synthetic basin (th-std 0.12, omega-std 0.5). But the energy-pump swing
controller delivers the poles to the top FAST and in varied configs that
sit outside that basin -> only ~15-19% of arrivals get held (see
swing-analytic-ceiling: the ceiling is the catcher's region of attraction,
not the swinger or DR).

Fix: reset a high fraction of balance episodes from states the energy pump
*actually* produces at the top, and let PPO learn nonlinear recovery from
them -- a basin matched to the swinger's output rather than a synthetic
gaussian. Warm-started from policy_uu.pt so it keeps the tight-basin hold
and extends it outward.

Usage:  python catch_train.py [--out policy_uu_catch.pt] [--iters 250]
Deploy: watch_math.py loads the up-up catcher; point it at policy_uu_catch.pt
        (or cp policy_uu_catch.pt policy_uu.pt after a good eval).
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from math_env import MathCartPoleVec
from train import Network, VecRunner, run, NUM_ENVS
from swing_energy import energy_command

ARRIVAL_MAX = 40000
UPPER_COS = 0.5           # "at the top" region where the catcher takes over


def collect_arrivals(n_states, seed=1):
    """Roll the energy-pump swing controller under full DR + wide starts;
    record the state each time both links first enter the upper region.
    These are exactly the states the deployed catcher must recover."""
    env = MathCartPoleVec(NUM_ENVS, seed=seed, fixed_goal=(1.0, 1.0),
                          balance_frac=0.0)
    armed = np.ones(NUM_ENVS, dtype=bool)   # can record once per top-visit
    out = []
    while len(out) < n_states:
        u = energy_command(env.theta1, env.omega1, env.theta2, env.omega2,
                           env.x, env.v)
        env.step(u)
        c1, c2 = np.cos(env.theta1), np.cos(env.theta2)
        up = (c1 > UPPER_COS) & (c2 > UPPER_COS)
        rec = up & armed
        for i in np.flatnonzero(rec):
            out.append([env.x[i], env.v[i], env.theta1[i], env.omega1[i],
                        env.theta2[i], env.omega2[i]])
        armed[rec] = False           # re-arm only after it leaves the top
        armed[~up] = True
    return np.array(out[:n_states], dtype=np.float64)


class ArrivalCartPole(MathCartPoleVec):
    """Same physics + domain randomization as the base env, but the
    'balance' fraction of resets is drawn from real swing-arrival states
    (plus light noise) instead of the narrow synthetic gaussian."""

    def __init__(self, *a, arrivals=None, **k):
        self.arrivals = arrivals
        self._arng = np.random.default_rng(12345)
        super().__init__(*a, **k)

    def _randomize(self, idx):
        super()._randomize(idx)               # DR params, goal, switch_in
        if self.arrivals is None:
            return
        idx = np.atleast_1d(idx)
        k = idx.size
        use = self._arng.random(k) < self.balance_frac
        sel = self._arng.integers(len(self.arrivals), size=k)
        s = self.arrivals[sel]
        an = self._arng.normal(0.0, 0.05, (k, 2))   # angle jitter (th1, th2)
        on = self._arng.normal(0.0, 0.20, (k, 2))   # angvel jitter (w1, w2)
        for j in np.flatnonzero(use):
            i = idx[j]
            self.x[i] = np.clip(s[j, 0], -0.9 * 500.0, 0.9 * 500.0)
            self.v[i] = s[j, 1]
            self.theta1[i] = s[j, 2] + an[j, 0]
            self.omega1[i] = s[j, 3] + on[j, 0]
            self.theta2[i] = s[j, 4] + an[j, 1]
            self.omega2[i] = s[j, 5] + on[j, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="policy_uu_catch.pt")
    ap.add_argument("--init", default="policy_uu.pt",
                    help="warm-start catcher (keeps the tight-basin hold)")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--balance-frac", type=float, default=0.9)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--std-init", type=float, default=0.08)
    ap.add_argument("--std-min", type=float, default=0.05)
    ap.add_argument("--critic-warmup", type=int, default=5)
    args = ap.parse_args()

    print(f"collecting {ARRIVAL_MAX} energy-pump arrival states...")
    arrivals = collect_arrivals(ARRIVAL_MAX)
    print(f"arrivals: mean|th1|={np.abs(arrivals[:,2]).mean():.2f} "
          f"mean|w1|={np.abs(arrivals[:,3]).mean():.2f} "
          f"mean|w2|={np.abs(arrivals[:,5]).mean():.2f}")

    net = Network(args.std_init)
    if args.init and os.path.exists(args.init):
        net.load_state_dict(torch.load(args.init, weights_only=True))
        net.log_std.data.fill_(float(np.log(args.std_init)))
        print(f"warm-started from {args.init}")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = ArrivalCartPole(NUM_ENVS, fixed_goal=(1.0, 1.0),
                          balance_frac=args.balance_frac, arrivals=arrivals)
    runner = VecRunner(env, max_steps=args.max_steps)

    log = open("training_log_uu_catch.csv", "w")
    log.write("iter,avg_reward,avg_len,episodes,std,loss,"
              "score_uu,score_ud,score_du,score_dd\n")
    try:
        run(net, optimizer, runner, log, args.iters, args.out,
            lr=args.lr, std_min=args.std_min,
            critic_warmup=args.critic_warmup)
    except KeyboardInterrupt:
        print(f"\ninterrupted — saving {args.out}")
    log.close()
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
