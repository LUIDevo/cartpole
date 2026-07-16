"""Swing-up via trajectory optimization, distilled into the policy net.

PPO cannot discover the swing-up->catch maneuver by exploration (six
reward/curriculum variants topped out at ~8% per-attempt success). Same
lesson as the balance skill: compute the controller, then clone it.

Pipeline:
  1. iLQR on the exact simulator dynamics (nominal params) finds a
     swing-up trajectory from a given start to upright, respecting
     force limits and track bounds.
  2. The iLQR backward pass yields time-varying feedback gains (TVLQR);
     the closed-loop controller is rolled out across the domain
     randomization range to generate a state->command dataset.
  3. The dataset is cloned into the actor (like distill.py); PPO can
     fine-tune afterwards with train.py --handoff.

Usage: python swing_opt.py [--starts 120] [--out policy_uu_swing.pt]
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from math_env import (MathCartPoleVec, DT, GRAVITY, POLE_LEN, R_COM,
                      I_COM_PER_M, HALF_TRACK, MAX_CART_VEL, ANGULAR_DAMP,
                      OMEGA_MAX, HANDOFF_COS, HANDOFF_OMEGA)
from train import Network

# nominal plant (mid-range of the DR draws)
M1 = M2 = 3.0
CART_MASS = 5.5
POLE_MASS_BELIEF = 2.5
MAX_FORCE = 24000.0
MAX_POWER = 7000.0

TOTAL = CART_MASS + POLE_MASS_BELIEF
MAX_SPEED = min(MAX_POWER / (TOTAL * 0.5), MAX_CART_VEL)
L, R, G, IC = POLE_LEN, R_COM, GRAVITY, I_COM_PER_M

T_HORIZON = 300          # 5 s
N_STATE, N_CTRL = 6, 1

# cost weights
CU = 0.02                # control effort
CX_BARRIER = 40.0        # track-bound barrier (per unit^2 beyond margin)
BARRIER_AT = 380.0
TERM_ANG = 600.0
TERM_OMG = 8.0
TERM_V = 2e-4
TERM_X = 0.5


def f(z, u):
    """One simulator step, nominal params. z=(x,v,th1,w1,th2,w2), u in [-1,1]."""
    x, v, th1, w1, th2, w2 = z
    u = np.clip(u, -1.0, 1.0)
    accel = u * MAX_FORCE / TOTAL
    v_new = np.clip(v + accel * DT, -MAX_SPEED, MAX_SPEED)
    x_new = x + v_new * DT
    a_piv = (v_new - v) / DT

    s1, c1 = np.sin(th1), np.cos(th1)
    s2, c2 = np.sin(th2), np.cos(th2)
    sd, cd = np.sin(th1 - th2), np.cos(th1 - th2)
    m11 = IC * M1 + M1 * R * R + M2 * L * L
    m22 = IC * M2 + M2 * R * R
    a_c = M2 * L * R
    m12 = a_c * cd
    h = M1 * R + M2 * L
    rhs1 = h * (G * s1 - a_piv * c1) - a_c * sd * w2 * w2
    rhs2 = M2 * R * (G * s2 - a_piv * c2) + a_c * sd * w1 * w1
    det = m11 * m22 - m12 * m12
    al1 = (rhs1 * m22 - m12 * rhs2) / det
    al2 = (rhs2 * m11 - m12 * rhs1) / det
    damp = 1.0 - ANGULAR_DAMP * DT
    w1n = np.clip((w1 + al1 * DT) * damp, -OMEGA_MAX, OMEGA_MAX)
    w2n = np.clip((w2 + al2 * DT) * damp, -OMEGA_MAX, OMEGA_MAX)
    return np.array([x_new, v_new, th1 + w1n * DT, w1n, th2 + w2n * DT, w2n])


def stage_cost(z, u):
    b = max(0.0, abs(z[0]) - BARRIER_AT)
    return CU * u * u + CX_BARRIER * (b / 100.0) ** 2


def term_cost(z):
    return (TERM_ANG * ((1 - np.cos(z[2])) + (1 - np.cos(z[4])))
            + TERM_OMG * (z[3] ** 2 + z[5] ** 2)
            + TERM_V * z[1] ** 2 + TERM_X * (z[0] / HALF_TRACK) ** 2)


def _jac(fun, z, u, eps=1e-5):
    fz = np.zeros((N_STATE, N_STATE))
    for i in range(N_STATE):
        d = np.zeros(N_STATE); d[i] = eps
        fz[:, i] = (fun(z + d, u) - fun(z - d, u)) / (2 * eps)
    fu = (fun(z, u + eps) - fun(z, u - eps)) / (2 * eps)
    return fz, fu.reshape(N_STATE, 1)


def _quad(cost, z, u, eps=1e-4):
    """Gradient+GN Hessian of stage cost via finite differences."""
    gz = np.zeros(N_STATE)
    for i in range(N_STATE):
        d = np.zeros(N_STATE); d[i] = eps
        gz[i] = (cost(z + d, u) - cost(z - d, u)) / (2 * eps)
    gu = (cost(z, u + eps) - cost(z, u - eps)) / (2 * eps)
    hz = np.zeros((N_STATE, N_STATE))
    for i in range(N_STATE):
        d = np.zeros(N_STATE); d[i] = eps
        hz[i, i] = (cost(z + d, u) - 2 * cost(z, u) + cost(z - d, u)) / eps ** 2
    hu = (cost(z, u + eps) - 2 * cost(z, u) + cost(z, u - eps)) / eps ** 2
    return gz, gu, np.maximum(hz, 0.0), max(hu, 1e-6)


def ilqr(z0, iters=120, verbose=False):
    """Returns (Z, U, K, ok). K: (T, 1, 6) feedback gains."""
    U = 0.001 * np.random.default_rng(0).standard_normal(T_HORIZON)
    Z = np.zeros((T_HORIZON + 1, N_STATE))
    Z[0] = z0
    for t in range(T_HORIZON):
        Z[t + 1] = f(Z[t], U[t])
    cost = sum(stage_cost(Z[t], U[t]) for t in range(T_HORIZON)) + term_cost(Z[-1])

    mu = 1.0
    K = np.zeros((T_HORIZON, 1, N_STATE))
    for it in range(iters):
        # backward pass
        gz = np.zeros(N_STATE)
        for i in range(N_STATE):
            d = np.zeros(N_STATE); d[i] = 1e-4
            gz[i] = (term_cost(Z[-1] + d) - term_cost(Z[-1] - d)) / 2e-4
        Vz = gz
        Vzz = np.zeros((N_STATE, N_STATE))
        for i in range(N_STATE):
            d = np.zeros(N_STATE); d[i] = 1e-4
            Vzz[i, i] = (term_cost(Z[-1] + d) - 2 * term_cost(Z[-1])
                         + term_cost(Z[-1] - d)) / 1e-8
        kff = np.zeros(T_HORIZON)
        diverged = False
        for t in reversed(range(T_HORIZON)):
            fz, fu = _jac(f, Z[t], U[t])
            cz, cu_, czz, cuu = _quad(stage_cost, Z[t], U[t])
            Qz = cz + fz.T @ Vz
            Qu = cu_ + (fu.T @ Vz)[0]
            Qzz = czz + fz.T @ Vzz @ fz
            Quu = cuu + (fu.T @ (Vzz + mu * np.eye(N_STATE)) @ fu)[0, 0]
            Quz = (fu.T @ (Vzz + mu * np.eye(N_STATE)) @ fz)[0]
            if Quu <= 1e-9:
                diverged = True
                break
            kff[t] = -Qu / Quu
            K[t, 0] = -Quz / Quu
            Vz = Qz + K[t, 0] * Qu + Quu * kff[t] * K[t, 0] + Quz * kff[t]
            Vzz = (Qzz + np.outer(K[t, 0], Quz) + np.outer(Quz, K[t, 0])
                   + Quu * np.outer(K[t, 0], K[t, 0]))
            Vzz = 0.5 * (Vzz + Vzz.T)
        if diverged:
            mu *= 4.0
            continue

        # forward line search
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.03):
            Zn = np.zeros_like(Z); Un = np.zeros_like(U)
            Zn[0] = z0
            for t in range(T_HORIZON):
                Un[t] = np.clip(U[t] + alpha * kff[t]
                                + K[t, 0] @ (Zn[t] - Z[t]), -1.0, 1.0)
                Zn[t + 1] = f(Zn[t], Un[t])
            cn = sum(stage_cost(Zn[t], Un[t]) for t in range(T_HORIZON)) \
                + term_cost(Zn[-1])
            if cn < cost - 1e-6:
                Z, U, cost = Zn, Un, cn
                improved = True
                break
        if improved:
            mu = max(mu / 2.0, 1e-6)
        else:
            mu *= 4.0
            if mu > 1e8:
                break
        if verbose and it % 20 == 0:
            print(f"    ilqr it {it:3d} cost {cost:10.2f} mu {mu:.1e}")

    zf = Z[-1]
    ok = ((1 - np.cos(zf[2])) < 0.05 and (1 - np.cos(zf[4])) < 0.05
          and abs(zf[3]) < 1.2 and abs(zf[5]) < 1.2 and abs(zf[0]) < 450)
    return Z, U, K, ok


def motor_invert(u, env, i):
    """Command that produces desired nominal-force fraction u on env i."""
    mag = np.abs(u) ** (1.0 / env.exponent[i])
    mag = mag * (1.0 - env.deadzone[i]) + (env.deadzone[i] if mag > 0 else 0.0)
    return np.sign(u) * mag - env.bias[i]


def collect_closed_loop(Z, U, K, n_dr, rng):
    """Roll the TVLQR controller in the real env across DR draws.
    Returns (obs, cmd) pairs from runs that end caught."""
    obs_out, cmd_out = [], []
    env = MathCartPoleVec(n_dr, seed=int(rng.integers(1 << 30)),
                          fixed_goal=(1.0, 1.0))
    env.x[:] = Z[0][0]; env.v[:] = Z[0][1]
    env.theta1[:] = Z[0][2]; env.omega1[:] = Z[0][3]
    env.theta2[:] = Z[0][4]; env.omega2[:] = Z[0][5]
    obs_run = [[] for _ in range(n_dr)]
    cmd_run = [[] for _ in range(n_dr)]
    caught = np.zeros(n_dr, dtype=bool)
    for t in range(T_HORIZON):
        z_ref, u_ref = Z[t], U[t]
        ob = env.observe()
        cmds = np.zeros(n_dr)
        for i in range(n_dr):
            if caught[i]:
                continue
            zi = np.array([env.x[i], env.v[i], env.theta1[i], env.omega1[i],
                           env.theta2[i], env.omega2[i]])
            dz = zi - z_ref
            # angle differences wrapped
            dz[2] = np.arctan2(np.sin(dz[2]), np.cos(dz[2]))
            dz[4] = np.arctan2(np.sin(dz[4]), np.cos(dz[4]))
            u = float(np.clip(u_ref + K[t, 0] @ dz, -1.0, 1.0))
            # scale force fraction for this env's motor strength/mass
            u = u * (MAX_FORCE / TOTAL) / (env.max_force[i]
                                           / (env.cart_mass[i]
                                              + env.pole_mass_belief[i]))
            u = float(np.clip(u, -1.0, 1.0))
            cmds[i] = motor_invert(u, env, i)
            obs_run[i].append(ob[i])
            cmd_run[i].append(np.float32(cmds[i]))
        env.step(cmds)
        c1, c2 = np.cos(env.theta1), np.cos(env.theta2)
        now = (~caught & (c1 > HANDOFF_COS) & (c2 > HANDOFF_COS)
               & (np.abs(env.omega1) < HANDOFF_OMEGA)
               & (np.abs(env.omega2) < HANDOFF_OMEGA))
        caught |= now
        if caught.all():
            break
    for i in range(n_dr):
        if caught[i]:
            obs_out.extend(obs_run[i])
            cmd_out.extend(cmd_run[i])
    return obs_out, cmd_out, int(caught.sum())


def sample_start(rng):
    th = lambda: (rng.uniform(-np.pi, np.pi) if rng.random() < 0.5
                  else rng.normal(0.0, 0.3))
    return np.array([rng.uniform(-250, 250), rng.uniform(-60, 60),
                     th(), rng.uniform(-2, 2), th(), rng.uniform(-2, 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=120)
    ap.add_argument("--dr-per-traj", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="policy_uu_swing.pt")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    all_obs, all_cmd = [], []
    n_ok = n_caught = n_total = 0
    for s in range(args.starts):
        z0 = (np.array([0.0, 0.0, np.pi, 0.0, np.pi, 0.0]) if s == 0
              else sample_start(rng))
        Z, U, K, ok = ilqr(z0)
        if not ok:
            continue
        n_ok += 1
        o, c, k = collect_closed_loop(Z, U, K, args.dr_per_traj, rng)
        all_obs.extend(o)
        all_cmd.extend(c)
        n_caught += k
        n_total += args.dr_per_traj
        if s % 10 == 0:
            print(f"start {s:3d}: trajectories ok {n_ok}, closed-loop catch "
                  f"{n_caught}/{n_total}, dataset {len(all_obs)}", flush=True)

    print(f"iLQR converged {n_ok}/{args.starts}; TVLQR catch rate "
          f"{100 * n_caught / max(1, n_total):.0f}%; "
          f"dataset {len(all_obs)} samples")
    if len(all_obs) < 5000:
        raise SystemExit("dataset too small — optimizer failing, abort")

    X = torch.from_numpy(np.stack(all_obs))
    Y = torch.tensor(all_cmd)
    net = Network()
    opt = torch.optim.Adam(net.actor.parameters(), lr=1e-3)
    B = 8192
    for ep in range(args.epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for kk in range(0, len(X), B):
            mb = perm[kk:kk + B]
            loss = ((net(X[mb]) - Y[mb]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(mb)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"bc epoch {ep:2d}  mse {tot / len(X):.5f}", flush=True)
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
