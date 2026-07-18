#!/usr/bin/env python3
"""Analytic energy-pumping swing-up for the cart-double-pole.

No training. The cart's acceleration a_pivot couples into the pendulum
energy through the generalized torques Q1 = -h*a_pivot*c1 and
Q2 = -m2*r*a_pivot*c2 (see math_env.step), so the power delivered to the
two links is

    dE/dt = -a_pivot * G,   G = h*c1*w1 + m2*r*c2*w2,   h = m1*r + m2*L.

To drive the pendulum energy E toward the upright rest energy e_up we
therefore command

    a_pivot = kE * (E - e_up) * sign(G)

which makes dE/dt = -kE*(E-e_up)*|G| -> pushes E to e_up. A light
centering term keeps the cart off the walls. Runtime hands off to the
balance net (policy_uu.pt) the instant both links enter the catch basin;
this module only has to *deliver* a catchable state.

Run `python swing_energy.py` for an honest per-attempt eval identical to
the iLQR pipeline's Stage B.
"""
import numpy as np

from math_env import POLE_LEN, R_COM, I_COM_PER_M, GRAVITY, HALF_TRACK

# nominal plant the controller believes in (midpoints of the DR ranges);
# energy pumping is gain-robust, so blindness to the true draw is fine
NM1 = NM2 = 3.0
N_CART = 5.0
N_FORCE = 24000.0

L, r = POLE_LEN, R_COM


def energy_command(theta1, omega1, theta2, omega2, x, v,
                   kE=6e-4, k_x=4.0, k_v=1.5, eps=0.5, a_max=8000.0):
    """Vectorised energy-pump command in [-1, 1] for a batch of states."""
    c1, c2 = np.cos(theta1), np.cos(theta2)

    m11 = I_COM_PER_M * NM1 + NM1 * r * r + NM2 * L * L
    m22 = I_COM_PER_M * NM2 + NM2 * r * r
    m12 = NM2 * L * r * np.cos(theta1 - theta2)
    ke = 0.5 * (m11 * omega1 ** 2 + 2.0 * m12 * omega1 * omega2
                + m22 * omega2 ** 2)
    pe = GRAVITY * (NM1 * r * c1 + NM2 * (L * c1 + r * c2))
    e_up = GRAVITY * (NM1 * r + NM2 * (L + r))

    h = NM1 * r + NM2 * L
    G = h * c1 * omega1 + NM2 * r * c2 * omega2
    # smooth sign to avoid chatter through G = 0
    a_pump = kE * (ke + pe - e_up) * np.tanh(G / eps)

    # keep the cart centred so it does not park on a wall mid-swing
    a_center = -k_x * (x / HALF_TRACK) * 100.0 - k_v * v
    a_cmd = np.clip(a_pump + a_center, -a_max, a_max)

    total_mass = N_CART + NM1 + NM2
    u = a_cmd * total_mass / N_FORCE
    return np.clip(u, -1.0, 1.0)


def _eval():
    import torch
    from math_env import HANDOFF_COS, HANDOFF_OMEGA
    from train import Network

    RELEASE_COS = 0.7

    def load(p):
        n = Network()
        n.load_state_dict(torch.load(p, weights_only=True))
        n.eval()
        return n

    bal = load("policy_uu.pt")
    N = 512
    env = MathCartPoleVec(N, seed=17, fixed_goal=(1.0, 1.0), balance_frac=0.0)
    mode = np.zeros(N, dtype=bool)          # False=swing, True=balance
    handoff_at = np.full(N, -1.0)
    up_ticks = np.zeros(N)
    for t in range(1800):                   # 30 s at 60 Hz
        u_swing = energy_command(env.theta1, env.omega1, env.theta2,
                                 env.omega2, env.x, env.v)
        obs = torch.from_numpy(env.observe())
        with torch.no_grad():
            u_bal = bal(obs).numpy()
        u = np.where(mode, u_bal, u_swing)
        env.step(u)
        c1, c2 = np.cos(env.theta1), np.cos(env.theta2)
        enter = ~mode & (c1 > HANDOFF_COS) & (c2 > HANDOFF_COS) \
            & (np.abs(env.omega1) < HANDOFF_OMEGA) \
            & (np.abs(env.omega2) < HANDOFF_OMEGA)
        mode |= enter
        handoff_at[enter & (handoff_at < 0)] = t / 60.0
        mode &= ~(mode & ((c1 < RELEASE_COS) | (c2 < RELEASE_COS)))
        up_ticks += (c1 > 0.9) & (c2 > 0.9)

    caught = handoff_at >= 0
    print(f"per-attempt handoff success: {100*caught.mean():.0f}%  "
          f"median t {np.median(handoff_at[caught]) if caught.any() else -1:.1f}s")
    print(f"mean both-up: {up_ticks.mean()/60:.1f}s/30s   "
          f">15s up: {100*(up_ticks/60 > 15).mean():.0f}%")


if __name__ == "__main__":
    from math_env import MathCartPoleVec
    _eval()
