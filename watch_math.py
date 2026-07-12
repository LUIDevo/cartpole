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

from math_env import MathCartPoleVec, HALF_TRACK, POLE_LEN
from train import Network, MAX_STEPS

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

    net = Network()
    try:
        net.load_state_dict(torch.load(args.policy, weights_only=True))
        net.eval()
        policy_name = args.policy
    except (FileNotFoundError, RuntimeError) as e:
        net = None
        policy_name = f"none ({e.__class__.__name__}: train first)"

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
             "goal": (1.0, 1.0)}

    def do_reset(_event=None):
        env.reset_all()
        env.set_goal(0, *state["goal"])
        state["obs"] = env.observe()
        state["steps"] = 0
        state["reward"] = 0.0
        state["episode"] += 1

    root.bind("<KeyPress-r>", do_reset)

    goals = {"1": (1.0, 1.0), "2": (1.0, -1.0), "3": (-1.0, 1.0), "4": (-1.0, -1.0)}
    goal_names = {(1.0, 1.0): "UP-UP", (1.0, -1.0): "UP-DOWN",
                  (-1.0, 1.0): "DOWN-UP", (-1.0, -1.0): "DOWN-DOWN"}

    def set_goal(a, b):
        state["goal"] = (a, b)
        env.set_goal(0, a, b)
        state["obs"] = env.observe()

    for key, (a, b) in goals.items():
        root.bind(f"<KeyPress-{key}>", lambda e, a=a, b=b: set_goal(a, b))

    def tick():
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
