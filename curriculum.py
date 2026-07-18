#!/usr/bin/env python3
"""Reverse-curriculum swing-up: ONE policy, from up-up outward to hanging.

The whole idea in one place. Instead of an energy-pump swinger + a
separate balance catcher + a distillation step (all dead ends — see the
swing-analytic-ceiling finding), a single policy is trained by growing
the set of START states outward from the goal:

  1. Start only at up-up-at-rest. The warm-started net already holds it.
  2. Whenever an episode HOLDS up-up, archive the states it passed through
     on the way in (its approach corridor). Those are, by construction,
     states from which the policy can already reach the goal -> they
     become the next, slightly-harder start states.
  3. The start frontier therefore marches outward from up toward down
     ONLY along trajectories the policy already solves, so every start
     has a dense gradient. No isotropic blob of unreachable states.

Episodes end only on a wall or timeout -- never for leaving the top -- so
the policy is free to swing down and pump energy for a bigger upswing.

The `frontier` (how far from upright the buffer reaches, 0..2pi) is logged
every iter, so the curriculum's progress toward full swing-up is
measurable, not hoped for.

Usage:  python curriculum.py [--init policy_uu.pt] [--iters 1500]
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from math_env import (MathCartPoleVec, HALF_TRACK, HIST_LEN, SEED_OFFSETS,
                      SEED_NOISE_ANG, SEED_NOISE_OMG)
from train import (Network, VecRunner, update, STEPS_PER_ITER, NUM_ENVS,
                   GOAL_NAMES, LR, MIN_STD)

SUCCESS_TICKS = 120        # held up-up this many ticks (2 s) => solvable
ANCHOR_FRAC = 0.3          # resets kept at pure up-up to retain the hold
BUF_MAX = 50000
UP_COS = 0.9


class CurriculumCartPole(MathCartPoleVec):
    def __init__(self, n, **k):
        self._ready = False
        super().__init__(n, fixed_goal=(1.0, 1.0), goal_switching=False, **k)
        self._hist = np.zeros((HIST_LEN, n, 6))
        self._age = np.zeros(n, dtype=np.int64)
        self._up = np.zeros(n, dtype=np.int64)
        self._ptr = 0
        self._buffer = [np.zeros(6)]          # [th1, w1, th2, w2, x, v] = up-rest
        self._arng = np.random.default_rng(999)
        self._ready = True
        self.reset_all()                      # place every env at the anchor

    def frontier(self):
        b = np.array(self._buffer)
        th1 = np.abs(np.arctan2(np.sin(b[:, 0]), np.cos(b[:, 0])))
        th2 = np.abs(np.arctan2(np.sin(b[:, 2]), np.cos(b[:, 2])))
        return float((th1 + th2).max())       # 0 (up) .. 2*pi (both hanging)

    def step(self, commands):
        obs, rew, done = super().step(commands)   # non-handoff: done = wall|bad
        c1, c2 = np.cos(self.theta1), np.cos(self.theta2)
        self._hist[self._ptr] = np.stack(
            [self.theta1, self.omega1, self.theta2, self.omega2,
             self.x, self.v], axis=1)
        self._age += 1
        self._up += (c1 > UP_COS) & (c2 > UP_COS)
        self._ptr = (self._ptr + 1) % HIST_LEN
        return obs, rew, done

    def _randomize(self, idx):
        super()._randomize(idx)               # DR params + goal + switch_in
        if not self._ready:
            return                            # first call is from super().__init__
        idx = np.atleast_1d(idx)
        # archive approach corridors of episodes that held the goal
        for i in idx:
            if self._up[i] >= SUCCESS_TICKS:
                for o in SEED_OFFSETS[SEED_OFFSETS < self._age[i]]:
                    self._buffer.append(
                        self._hist[(self._ptr - o) % HIST_LEN, i].copy())
        if len(self._buffer) > BUF_MAX:
            del self._buffer[:len(self._buffer) - BUF_MAX]
        # draw new starts: anchor, or a solved buffer state + light noise
        k = idx.size
        anchor = (self._arng.random(k) < ANCHOR_FRAC)
        sel = self._arng.integers(len(self._buffer), size=k)
        an = self._arng.normal(0.0, SEED_NOISE_ANG, (k, 2))
        on = self._arng.normal(0.0, SEED_NOISE_OMG, (k, 2))
        for j, i in enumerate(idx):
            if anchor[j]:
                s = np.zeros(6)
            else:
                s = self._buffer[sel[j]]
            self.theta1[i] = s[0] + an[j, 0]
            self.omega1[i] = s[1] + on[j, 0]
            self.theta2[i] = s[2] + an[j, 1]
            self.omega2[i] = s[3] + on[j, 1]
            self.x[i] = np.clip(s[4], -0.9 * HALF_TRACK, 0.9 * HALF_TRACK)
            self.v[i] = s[5]
        self._age[idx] = 0
        self._up[idx] = 0


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
