#!/usr/bin/env python3
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sim_env import SimEnv
from train import Network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="policy.pt", help="checkpoint to load")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--episodes", type=int, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions with exploration noise (training behavior)")
    args = ap.parse_args()

    net = Network()
    net.load_state_dict(torch.load(args.policy, weights_only=True))
    net.eval()
    print(f"loaded {args.policy}"
          + (" (stochastic)" if args.stochastic else " (deterministic)"))
    print("focus the sim window: hold Left/Right arrows to shove the cart, "
          "release to let the policy recover")

    episode = 0
    with SimEnv(port=args.port, headless=False) as sim:
        try:
            while args.episodes == 0 or episode < args.episodes:
                state = torch.tensor(sim.reset(), dtype=torch.float32)
                total_reward, steps, done = 0.0, 0, False
                while not done:
                    with torch.no_grad():
                        if args.stochastic:
                            command = float(net.dist(state).sample().clamp(-1.0, 1.0))
                        else:
                            command = float(net(state))
                    next_state, reward, done = sim.step(command)
                    total_reward += reward
                    steps += 1
                    state = torch.tensor(next_state, dtype=torch.float32)
                episode += 1
                print(f"episode {episode:3d}  reward {total_reward:8.2f}  steps {steps:5d}")
        except KeyboardInterrupt:
            pass

    print(f"watched {episode} episode(s)")


if __name__ == "__main__":
    main()
