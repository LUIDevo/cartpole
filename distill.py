"""Distill an LQR balance teacher into the policy network.

PPO cannot discover the up-up balance by exploration: noise-driven
search never produces a single successful hold to reinforce, at any
noise level. A linear controller derived from the linearized double-
pendulum dynamics holds up-up even under full domain randomization
(~88% for 20 s), so we clone it into the actor and let PPO fine-tune
from there (see run_curriculum_uu.sh).

Usage: python distill.py [--out policy_uu_bc.pt]
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from math_env import (MathCartPoleVec, DT, GRAVITY, POLE_LEN, R_COM,
                      I_COM_PER_M)
from train import Network

NOMINAL_MASS = 3.0
SAMPLES = 1_000_000
EPOCHS = 30
BATCH = 8192
DITHER_STD = 0.05
NUM_ENVS = 512


def lqr_gains(m1=NOMINAL_MASS, m2=NOMINAL_MASS):
    """DLQR for the upright linearization; state (x, v, th1, w1, th2, w2),
    control = pivot acceleration. One nominal gain set stabilizes the
    whole randomization range."""
    L, r, g = POLE_LEN, R_COM, GRAVITY
    m11 = I_COM_PER_M * m1 + m1 * r * r + m2 * L * L
    m22 = I_COM_PER_M * m2 + m2 * r * r
    m12 = m2 * L * r
    minv = np.linalg.inv(np.array([[m11, m12], [m12, m22]]))
    h = m1 * r + m2 * L

    A = np.zeros((6, 6))
    B = np.zeros((6, 1))
    A[0, 1] = A[2, 3] = A[4, 5] = 1.0
    ga = minv @ np.diag([h * g, m2 * r * g])
    ba = minv @ -np.array([[h], [m2 * r]])
    A[3, [2, 4]] = ga[0]
    A[5, [2, 4]] = ga[1]
    B[1, 0] = 1.0
    B[3, 0] = ba[0, 0]
    B[5, 0] = ba[1, 0]

    Ad = np.eye(6) + A * DT
    Bd = B * DT
    Q = np.diag([1e-4, 1e-4, 10.0, 0.1, 10.0, 0.1])
    R = np.array([[1e-6]])
    P = Q.copy()
    for _ in range(20000):
        K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        Pn = Q + Ad.T @ P @ (Ad - Bd @ K)
        if np.abs(Pn - P).max() < 1e-9:
            P = Pn
            break
        P = Pn
    return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad).ravel()


def teacher_u(env, K):
    th1 = np.arctan2(np.sin(env.theta1), np.cos(env.theta1))
    th2 = np.arctan2(np.sin(env.theta2), np.cos(env.theta2))
    s = np.stack([env.x, env.v, th1, env.omega1, th2, env.omega2])
    a_des = -(K @ s)
    f_des = a_des * (env.cart_mass + env.pole_mass_belief)
    u = np.clip(f_des / env.max_force, -1.0, 1.0)
    # invert the motor nonlinearity (env-side params, teacher may cheat)
    mag = np.abs(u) ** (1.0 / env.exponent)
    mag = mag * (1.0 - env.deadzone) + np.where(mag > 0.0, env.deadzone, 0.0)
    return np.sign(u) * mag - env.bias


def basin_reset(env, rng, idx):
    n = len(idx)
    env.x[idx] = rng.uniform(-200, 200, n)
    env.v[idx] = rng.normal(0, 30, n)
    env.theta1[idx] = rng.normal(0.0, 0.12, n)
    env.theta2[idx] = rng.normal(0.0, 0.12, n)
    env.omega1[idx] = rng.normal(0.0, 0.5, n)
    env.omega2[idx] = rng.normal(0.0, 0.5, n)


def collect(samples):
    rng = np.random.default_rng(0)
    env = MathCartPoleVec(NUM_ENVS, seed=0, fixed_goal=(1.0, 1.0))
    K = lqr_gains()
    basin_reset(env, rng, np.arange(NUM_ENVS))
    obs_buf, act_buf = [], []
    for _ in range(samples // NUM_ENVS):
        u = teacher_u(env, K)
        obs_buf.append(env.observe())
        act_buf.append(u.astype(np.float32))
        env.step(np.clip(u + rng.normal(0, DITHER_STD, NUM_ENVS), -1, 1))
        fallen = ((np.cos(env.theta1) < 0.5) | (np.cos(env.theta2) < 0.5)
                  | (np.abs(env.x) > 450))
        if fallen.any():
            idx = np.flatnonzero(fallen)
            env._randomize(idx)  # re-roll the DR params too
            basin_reset(env, rng, idx)
    return (torch.from_numpy(np.concatenate(obs_buf)),
            torch.from_numpy(np.concatenate(act_buf)))


def hold_rate(net, seconds=20):
    env = MathCartPoleVec(256, seed=42, fixed_goal=(1.0, 1.0))
    rng = np.random.default_rng(1)
    env.x[:] = rng.uniform(-100, 100, 256)
    env.v[:] = 0.0
    env.theta1[:] = rng.normal(0, 0.05, 256)
    env.theta2[:] = rng.normal(0, 0.05, 256)
    env.omega1[:] = rng.normal(0, 0.2, 256)
    env.omega2[:] = rng.normal(0, 0.2, 256)
    alive = np.ones(256, dtype=bool)
    for _ in range(60 * seconds):
        with torch.no_grad():
            a = net(torch.from_numpy(env.observe())).numpy()
        env.step(a)
        alive &= (np.cos(env.theta1) > 0.7) & (np.cos(env.theta2) > 0.7)
    return float(alive.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="policy_uu_bc.pt")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    X, Y = collect(args.samples)
    print(f"teacher dataset: {len(X)} samples")

    net = Network()
    opt = torch.optim.Adam(net.actor.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        total = 0.0
        for k in range(0, len(X), BATCH):
            mb = perm[k:k + BATCH]
            loss = ((net(X[mb]) - Y[mb]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(mb)
        print(f"epoch {epoch:2d}  mse {total / len(X):.5f}", flush=True)

    torch.save(net.state_dict(), args.out)
    net.eval()
    rate = hold_rate(net)
    print(f"saved {args.out}")
    print(f"hold check: {rate * 100:.1f}% still up-up after 20s "
          f"(full DR, deterministic); expect ~75%+")
    if rate < 0.5:
        raise SystemExit("distillation failed the hold check")


if __name__ == "__main__":
    main()
