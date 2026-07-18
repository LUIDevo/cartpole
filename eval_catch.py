#!/usr/bin/env python3
"""Honest per-attempt eval: energy-pump swing + a chosen catcher net.

Usage: python eval_catch.py [catcher.pt ...]   (default: both nets)
Compares held-last-15s so the arrival-trained catcher can be measured
against the original policy_uu.pt on the swinger's real arrivals.
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np, torch
from math_env import MathCartPoleVec, HANDOFF_COS, HANDOFF_OMEGA
from train import Network
from swing_energy import energy_command

RELEASE_COS = 0.7


def load(p):
    n = Network(); n.load_state_dict(torch.load(p, weights_only=True))
    n.eval(); return n


def run(catcher, C_ON=HANDOFF_COS, W_ON=HANDOFF_OMEGA):
    N = 512
    env = MathCartPoleVec(N, seed=17, fixed_goal=(1.0, 1.0), balance_frac=0.0)
    mode = np.zeros(N, bool); hoff = np.full(N, -1.0); up_tail = np.zeros(N)
    for t in range(1800):
        us = energy_command(env.theta1, env.omega1, env.theta2, env.omega2,
                            env.x, env.v)
        with torch.no_grad():
            ub = catcher(torch.from_numpy(env.observe())).numpy()
        u = np.where(mode, ub, us); env.step(u)
        c1, c2 = np.cos(env.theta1), np.cos(env.theta2)
        enter = ~mode & (c1 > C_ON) & (c2 > C_ON) \
            & (np.abs(env.omega1) < W_ON) & (np.abs(env.omega2) < W_ON)
        mode |= enter; hoff[enter & (hoff < 0)] = t / 60
        mode &= ~(mode & ((c1 < RELEASE_COS) | (c2 < RELEASE_COS)))
        if t >= 900:
            up_tail += (c1 > 0.9) & (c2 > 0.9)
    return 100 * (hoff >= 0).mean(), 100 * (up_tail / 900 > 0.9).mean()


if __name__ == "__main__":
    paths = sys.argv[1:] or ["policy_uu.pt", "policy_uu_catch.pt"]
    for p in paths:
        if not os.path.exists(p):
            print(f"{p:22s}: (missing)"); continue
        h, u = run(load(p))
        print(f"{p:22s}: handoff {h:3.0f}%  held-last-15s {u:3.0f}%")
