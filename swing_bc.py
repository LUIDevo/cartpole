#!/usr/bin/env python3
"""Behaviour-clone the analytic energy-pump into the swing actor.

This is the warm-start for PPO --handoff. PPO cannot discover swing-up
by cold exploration (the six-collision credit-assignment problem noted
in swing_opt.py), so we clone the energy controller (swing_energy.py)
as a *prior*: it already delivers the poles to the top region, and PPO
then only has to learn the near-top configuration control that energy
shaping cannot provide (see the swing-analytic-ceiling finding).

Unlike the iLQR->BC route this dataset is single-mode: one deterministic
controller labels every state, so BC fits cleanly instead of averaging
conflicting trajectories to mush.

Usage:  python swing_bc.py [--out policy_uu_swing_bc.pt]
Then:   python train.py --goal uu --handoff --init policy_uu_swing_bc.pt \
                        --out policy_uu_swing.pt
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from math_env import MathCartPoleVec
from train import Network
from swing_energy import energy_command

NUM_ENVS = 512
SAMPLES = 1_000_000
EPOCHS = 30
BATCH = 8192
DITHER_STD = 0.05          # action noise for state-space coverage


def collect(samples):
    rng = np.random.default_rng(0)
    # full domain randomization, wide starts, up-up goal: the exact state
    # distribution the energy controller must cover for PPO to inherit it
    env = MathCartPoleVec(NUM_ENVS, seed=0, fixed_goal=(1.0, 1.0),
                          balance_frac=0.0)
    obs_buf, act_buf = [], []
    for _ in range(samples // NUM_ENVS):
        u = energy_command(env.theta1, env.omega1, env.theta2, env.omega2,
                           env.x, env.v)
        obs_buf.append(env.observe())
        act_buf.append(u.astype(np.float32))
        # dithered command for coverage; env re-randomizes on wall death
        env.step(np.clip(u + rng.normal(0, DITHER_STD, NUM_ENVS), -1, 1))
    return np.concatenate(obs_buf), np.concatenate(act_buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="policy_uu_swing_bc.pt")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    print(f"collecting {args.samples} energy-pump samples...")
    X, Y = collect(args.samples)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    net = Network()
    opt = torch.optim.Adam(net.actor.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for k in range(0, len(X), BATCH):
            mb = perm[k:k + BATCH]
            loss = ((net(X[mb]) - Y[mb]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(mb)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"bc epoch {ep:2d}  mse {tot / len(X):.5f}", flush=True)

    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")
    print("next: python train.py --goal uu --handoff "
          f"--init {args.out} --out policy_uu_swing.pt")


if __name__ == "__main__":
    main()
