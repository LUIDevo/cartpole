#!/usr/bin/env python3
"""Example neural-network control client for the cart-pole simulation.

Start the sim in control mode first:
    godot-mono --headless --path simulation -- --port=9999
Then run this client:
    python examples/nn_control_client.py --port 9999

Lockstep protocol (newline-delimited ASCII over TCP):
    sim -> client : "cart_velocity,pole_angular_velocity,pole_angle,done"
    client -> sim : "<command>"   float in [-1, 1]
                    "R"           reset the episode
"""
import argparse
import socket


def policy(cart_velocity, pole_angular_velocity, pole_angle):
    """Replace this with your neural network forward pass.

    Receives the observation, returns a motor command in [-1, 1].
    Placeholder below is a trivial hand-tuned balancer (push toward upright).
    """
    return max(-1.0, min(1.0, 3.0 * pole_angle + 0.5 * pole_angular_velocity))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--steps", type=int, default=2000, help="steps before disconnect")
    args = ap.parse_args()

    with socket.create_connection((args.host, args.port)) as sock:
        buf = ""
        for _ in range(args.steps):
            # read one observation line
            while "\n" not in buf:
                chunk = sock.recv(4096).decode("ascii")
                if not chunk:
                    return  # sim closed
                buf += chunk
            line, buf = buf.split("\n", 1)

            cart_v, pole_av, pole_a, done = line.split(",")
            cart_v, pole_av, pole_a, done = float(cart_v), float(pole_av), float(pole_a), int(done)

            if done:
                sock.sendall(b"R\n")   # pole fell -> reset episode
                continue

            u = policy(cart_v, pole_av, pole_a)
            sock.sendall(f"{u}\n".encode("ascii"))


if __name__ == "__main__":
    main()
