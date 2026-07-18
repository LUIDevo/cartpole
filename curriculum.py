#!/usr/bin/env python3
"""Reverse-curriculum swing-up: ONE policy, from up-up outward to hanging.

Instead of an energy-pump swinger + a separate balance catcher + a
distillation step (all dead ends -- see the swing-analytic-ceiling
finding), a single policy is trained by growing the set of START states
outward from the goal, along trajectories the policy already solves.

The mechanism (corrected after a first version poisoned itself):

  1. The buffer holds only PROVEN start states -- poses an episode was
     actually launched from AND then held up-up. The anchor (up-up at
     rest) seeds it. Nothing mid-episode is archived, so no flailing
     junk gets used as a start.
  2. Each new start is a proven buffer state + a small Gaussian
     perturbation (or, ANCHOR_FRAC of the time, the pure anchor). If the
     policy holds from the perturbed state, that state joins the buffer
     -- so the frontier ratchets outward by one perturbation step only
     where the policy actually succeeds. Over-reaching perturbations fail
     and are simply never added: no pollution.
  3. When the buffer is full, eviction is a tournament that preferentially
     drops states CLOSER to upright, so the buffer concentrates on the
     competence frontier; ANCHOR_FRAC keeps the up-up hold from being
     forgotten. The anchor itself is never evicted.
  4. Success is a real hold: the longest CONSECUTIVE up-up streak in the
     episode must exceed SUCCESS_TICKS (not total ticks, which flailing
     could fake).

`frontier` = 90th-percentile buffer tilt (0=up .. ~6.28=both hanging),
logged every iter so progress toward full swing-up is measurable.

Usage:  python curriculum.py [--init policy_uu.pt] [--iters 1500]
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from math_env import MathCartPoleVec, HALF_TRACK
from train import (Network, VecRunner, update, STEPS_PER_ITER, NUM_ENVS,
                   LR, MIN_STD)

SUCCESS_TICKS = 150        # longest consecutive up-up streak to count solved
ANCHOR_FRAC = 0.3          # resets kept at the pure anchor (retain the hold)
BUF_MAX = 40000
UP_COS = 0.9
PERT_ANG = 0.15            # per-generation outward step (angle, rad)
PERT_OMG = 0.30            # perturbation on angular velocity
PERT_X = 20.0              # perturbation on cart position
PERT_V = 10.0             # perturbation on cart velocity
ANCHOR = np.zeros(6)       # [th1, w1, th2, w2, x, v] = up-up at rest


def _tilt(states):
    """Total distance from upright, |wrap(th1)| + |wrap(th2)|."""
    s = np.atleast_2d(states)
    t1 = np.abs(np.arctan2(np.sin(s[:, 0]), np.cos(s[:, 0])))
    t2 = np.abs(np.arctan2(np.sin(s[:, 2]), np.cos(s[:, 2])))
    return t1 + t2


class CurriculumCartPole(MathCartPoleVec):
    def __init__(self, n, **k):
        self._ready = False
        super().__init__(n, fixed_goal=(1.0, 1.0), goal_switching=False, **k)
        self._buffer = [ANCHOR.copy()]        # proven start states
        self._start = np.zeros((n, 6))         # pose each env was launched from
        self._streak = np.zeros(n, dtype=np.int64)   # current up-up run
        self._best = np.zeros(n, dtype=np.int64)     # longest run this episode
        self._arng = np.random.default_rng(999)
        self._ready = True
        self.reset_all()                       # place every env via curriculum

    def frontier(self):
        return float(np.percentile(_tilt(np.array(self._buffer)), 90))

    def step(self, commands):
        obs, rew, done = super().step(commands)   # non-handoff: done = wall|bad
        up = (np.cos(self.theta1) > UP_COS) & (np.cos(self.theta2) > UP_COS)
        self._streak = np.where(up, self._streak + 1, 0)
        self._best = np.maximum(self._best, self._streak)
        return obs, rew, done

    def _evict(self):
        # tournament: drop the more-upright of two random non-anchor states,
        # concentrating the buffer on the frontier
        while len(self._buffer) > BUF_MAX:
            a, b = self._arng.integers(1, len(self._buffer), size=2)
            drop = a if _tilt(self._buffer[a])[0] < _tilt(self._buffer[b])[0] else b
            self._buffer.pop(int(drop))

    def _randomize(self, idx):
        super()._randomize(idx)               # DR params + goal + switch_in
        if not self._ready:
            return                            # first call is from super().__init__
        idx = np.atleast_1d(idx)
        # promote proven starts of episodes that actually held the goal
        for i in idx:
            if self._best[i] >= SUCCESS_TICKS:
                self._buffer.append(self._start[i].copy())
        self._evict()
        # draw new starts: anchor, or a proven state + small perturbation
        k = idx.size
        anchor = self._arng.random(k) < ANCHOR_FRAC
        sel = self._arng.integers(len(self._buffer), size=k)
        pert = self._arng.normal(0.0, 1.0, (k, 6)) * \
            np.array([PERT_ANG, PERT_OMG, PERT_ANG, PERT_OMG, PERT_X, PERT_V])
        for j, i in enumerate(idx):
            base = ANCHOR if anchor[j] else self._buffer[sel[j]]
            s = base + pert[j]
            s[4] = np.clip(s[4], -0.9 * HALF_TRACK, 0.9 * HALF_TRACK)
            self._start[i] = s
            self.theta1[i], self.omega1[i] = s[0], s[1]
            self.theta2[i], self.omega2[i] = s[2], s[3]
            self.x[i], self.v[i] = s[4], s[5]
        self._streak[idx] = 0
        self._best[idx] = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="policy_uu.pt",
                    help="warm-start (keeps the up-up hold; '' = from scratch)")
    ap.add_argument("--out", default="policy_uu_curriculum.pt")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--std-init", type=float, default=0.3)
    ap.add_argument("--std-min", type=float, default=MIN_STD)
    ap.add_argument("--max-steps", type=int, default=800)
    args = ap.parse_args()

    net = Network(args.std_init)
    if args.init and os.path.exists(args.init):
        net.load_state_dict(torch.load(args.init, weights_only=True))
        net.log_std.data.fill_(float(np.log(args.std_init)))
        print(f"warm-started from {args.init}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = CurriculumCartPole(NUM_ENVS)
    runner = VecRunner(env, max_steps=args.max_steps)

    log = open("training_log_curriculum.csv", "w")
    log.write("iter,avg_reward,episodes,std,loss,uu,frontier,buffer\n")
    for it in range(args.iters):
        frac = max(0.1, 1.0 - it / args.iters)
        for g in opt.param_groups:
            g["lr"] = args.lr * frac
        data, ep_r, ep_l, scores = runner.collect(net, STEPS_PER_ITER)
        loss = update(net, opt, *data, min_std=args.std_min)
        std = float(net.log_std.detach().exp())
        fr = env.frontier()
        avg = sum(ep_r) / len(ep_r) if ep_r else float("nan")
        print(f"iter {it:4d}  avg_reward {avg:8.2f}  eps {len(ep_l):3d}  "
              f"uu {scores['uu']:+.2f}  frontier {fr:4.2f}/6.28  "
              f"buffer {len(env._buffer):6d}  std {std:.3f}", flush=True)
        log.write(f"{it},{avg:.4f},{len(ep_l)},{std:.4f},{loss:.4f},"
                  f"{scores['uu']:.4f},{fr:.4f},{len(env._buffer)}\n")
        log.flush()
        if (it + 1) % 20 == 0:
            torch.save(net.state_dict(), args.out)
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
