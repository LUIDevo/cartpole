#!/usr/bin/env python3
"""Visualise the math simulation directly (no Godot).

Policy drives the cart; hold Left/Right arrows to shove it and release
to let the policy recover. Keys 1-4 command the goal (up-up, up-down,
down-up, down-down). R resets the episode, Escape quits.
"""
import argparse
import os
import tkinter as tk

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from math_env import (MathCartPoleVec, HALF_TRACK, POLE_LEN,
                      HANDOFF_COS, HANDOFF_OMEGA)
from train import Network, MAX_STEPS

# balance net hands back to the swing net only if knocked well out
RELEASE_COS = 0.7

WIDTH, HEIGHT = 1100, 560
SCALE = 0.9
CX, CY = WIDTH / 2, HEIGHT * 0.75
CART_W, CART_H = 44, 23


def pole_angles(env):
    if hasattr(env, "theta1"):
        return [float(env.theta1[0]), float(env.theta2[0])]
    return [float(env.theta[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="policy.pt")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    def load_net(path):
        n = Network()
        try:
            n.load_state_dict(torch.load(path, weights_only=True))
            n.eval()
            return n
        except (FileNotFoundError, RuntimeError):
            return None

    # specialist per goal; down-down needs no policy (motor off)
    specialists = {(1.0, 1.0): load_net("policy_uu.pt"),
                   (1.0, -1.0): load_net("policy_ud.pt"),
                   (-1.0, 1.0): load_net("policy_du.pt"),
                   (-1.0, -1.0): None}
    fallback = load_net(args.policy)
    swing_uu = load_net("policy_uu_swing.pt")
    loaded = [n for n in specialists.values() if n is not None]
    policy_name = (f"specialists ({len(loaded)}/3 loaded, dd = motor off"
                   + (", uu = swing+balance)" if swing_uu else ")")
                   if loaded else (args.policy if fallback else "none — train first"))

    env = MathCartPoleVec(1, goal_switching=False)
    obs = env.reset_all()
    env.set_goal(0, 1.0, 1.0)

    root = tk.Tk()
    root.title("cart-pole math sim")
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="#111318",
                       highlightthickness=0)
    canvas.pack()

    held = set()
    root.bind("<KeyPress-Left>", lambda e: held.add("L"))
    root.bind("<KeyRelease-Left>", lambda e: held.discard("L"))
    root.bind("<KeyPress-Right>", lambda e: held.add("R"))
    root.bind("<KeyRelease-Right>", lambda e: held.discard("R"))
    root.bind("<Escape>", lambda e: root.destroy())

    state = {"obs": obs, "steps": 0, "reward": 0.0, "episode": 1,
             "goal": (1.0, 1.0), "mode": "swing"}

    def do_reset(_event=None):
        env.reset_all()
        env.set_goal(0, *state["goal"])
        state["obs"] = env.observe()
        state["steps"] = 0
        state["reward"] = 0.0
        state["episode"] += 1
        state["mode"] = "swing"

    root.bind("<KeyPress-r>", do_reset)

    goals = {"1": (1.0, 1.0), "2": (1.0, -1.0), "3": (-1.0, 1.0), "4": (-1.0, -1.0)}
    goal_names = {(1.0, 1.0): "UP-UP", (1.0, -1.0): "UP-DOWN",
                  (-1.0, 1.0): "DOWN-UP", (-1.0, -1.0): "DOWN-DOWN"}

    def set_goal(a, b):
        state["goal"] = (a, b)
        env.set_goal(0, a, b)
        state["obs"] = env.observe()
        state["mode"] = "swing"

    for key, (a, b) in goals.items():
        root.bind(f"<KeyPress-{key}>", lambda e, a=a, b=b: set_goal(a, b))

    def tick():
        goal = state["goal"]
        net = specialists.get(goal)
        if net is None and goal != (-1.0, -1.0):
            net = fallback
        # up-up runs as two specialists: swing net delivers the poles
        # into the catch basin, balance net holds them there
        if goal == (1.0, 1.0) and swing_uu is not None and net is not None:
            c1, c2 = np.cos(float(env.theta1[0])), np.cos(float(env.theta2[0]))
            w1, w2 = float(env.omega1[0]), float(env.omega2[0])
            if state["mode"] == "swing" and (
                    c1 > HANDOFF_COS and c2 > HANDOFF_COS
                    and abs(w1) < HANDOFF_OMEGA and abs(w2) < HANDOFF_OMEGA):
                state["mode"] = "balance"
            elif state["mode"] == "balance" and (c1 < RELEASE_COS
                                                 or c2 < RELEASE_COS):
                state["mode"] = "swing"
            if state["mode"] == "swing":
                net = swing_uu
        if net is not None:
            with torch.no_grad():
                s = torch.from_numpy(state["obs"])
                if args.stochastic:
                    cmd = float(net.dist(s).sample().clamp(-1.0, 1.0)[0])
                else:
                    cmd = float(net(s)[0])
        else:
            cmd = 0.0
        if "L" in held and "R" not in held:
            cmd = -1.0
        elif "R" in held and "L" not in held:
            cmd = 1.0

        obs, rew, done = env.step(np.array([cmd]))
        state["obs"] = obs
        state["steps"] += 1
        state["reward"] += float(rew[0])

        draw(cmd)

        if bool(done[0]) or state["steps"] >= MAX_STEPS:
            do_reset()
        root.after(16, tick)

    def draw(cmd):
        canvas.delete("all")

        y_track = CY + CART_H / 2 + 4
        x_lo, x_hi = CX - HALF_TRACK * SCALE, CX + HALF_TRACK * SCALE
        canvas.create_line(x_lo, y_track, x_hi, y_track, fill="#8a8a94", width=3)
        for xw in (x_lo, x_hi):
            canvas.create_rectangle(xw - 4, y_track - 40, xw + 4, y_track + 5,
                                    fill="#c14b4b", outline="")

        x = CX + float(env.x[0]) * SCALE
        canvas.create_rectangle(x - CART_W / 2, CY - CART_H / 2,
                                x + CART_W / 2, CY + CART_H / 2,
                                fill="#4b77c1", outline="")

        px, py = x, CY
        colors = ["#e0e0e6", "#57d9c4"]
        for k, th in enumerate(pole_angles(env)):
            qx = px + POLE_LEN * SCALE * np.sin(th)
            qy = py - POLE_LEN * SCALE * np.cos(th)
            canvas.create_line(px, py, qx, qy, fill=colors[k % 2], width=8,
                               capstyle=tk.ROUND)
            canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#f2b134",
                               outline="")
            px, py = qx, qy

        manual = "L" in held or "R" in held
        goal = goal_names.get((float(env.g1[0]), float(env.g2[0])), "?")
        hud = (f"episode {state['episode']}   step {state['steps']:4d}   "
               f"reward {state['reward']:8.1f}   cmd {cmd:+.2f}   "
               f"goal {goal}"
               + (f"   [{state['mode'].upper()}]"
                  if goal == "UP-UP" and swing_uu is not None else "")
               + ("   [MANUAL]" if manual else ""))
        canvas.create_text(14, 16, anchor="w", fill="#d8d8de",
                           font=("monospace", 12), text=hud)
        canvas.create_text(14, 38, anchor="w", fill="#6f6f7a",
                           font=("monospace", 10),
                           text=f"policy: {policy_name}   keys: Left/Right shove, "
                                "1-4 goal (upup/updown/downup/downdown), R reset, Esc quit")

    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
