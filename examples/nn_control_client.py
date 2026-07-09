#!/usr/bin/env python3
"""Example neural-network control client for the cart-pole simulation.

Start the sim in control mode first:
    godot-mono --headless --path simulation -- --port=9999
Then run this client:
    python examples/nn_control_client.py --port 9999

Lockstep protocol (newline-delimited ASCII over TCP). Observations are
NORMALIZED to ~[-1, 1] (same scaling as the training CSV):
    cart_velocity         /= 500        (m/s-ish, sim units)
    pole_angular_velocity /= 10         (rad/s)
    pole_angle             = wrap(rad) / pi   (±180° -> ±1)
    cart_position          = |offset| / half_track   (0 -> 1 at track end)

    sim -> client : "cart_velocity,pole_angular_velocity,pole_angle,cart_position,reward,done"
    client -> sim : "<command>"   float in [-1, 1]   (nothing when done=1; the sim
                                                      auto-restarts the episode)

Every driven step is logged to a CSV (--out) with columns:
    episode_id,step,cart_velocity,pole_angular_velocity,pole_angle,cart_position,motor_command,reward,done
"""
import argparse
import csv
import socket


def policy(cart_velocity, pole_angular_velocity, pole_angle):
    """Replace this with your neural network forward pass.

    Receives the NORMALIZED observation, returns a motor command in [-1, 1].
    Placeholder below is a trivial hand-tuned balancer (push toward upright).
    Gains account for the normalized scale (angle /pi, ang-vel /10).
    """
    return max(-1.0, min(1.0, 9.4 * pole_angle + 5.0 * pole_angular_velocity))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--steps", type=int, default=2000, help="steps before disconnect")
    ap.add_argument("--out", default="data/control_log.csv", help="CSV output path")
    args = ap.parse_args()

    with socket.create_connection((args.host, args.port)) as sock, \
            open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_id", "step", "cart_velocity", "pole_angular_velocity",
            "pole_angle", "cart_position", "motor_command", "reward", "done"])

        episode_id, step = 0, 0
        buf = ""
        for _ in range(args.steps):
            # read one observation line
            while "\n" not in buf:
                chunk = sock.recv(4096).decode("ascii")
                if not chunk:
                    return  # sim closed
                buf += chunk
            line, buf = buf.split("\n", 1)

            cart_v, pole_av, pole_a, cart_p, reward, done = line.split(",")
            cart_v, pole_av, pole_a, cart_p, reward, done = (
                float(cart_v), float(pole_av), float(pole_a),
                float(cart_p), float(reward), int(done))

            if done:
                # log the terminal state (no command applied). The sim auto-restarts
                # on done, so we send nothing and just read the next episode's obs.
                writer.writerow([episode_id, step, cart_v, pole_av, pole_a,
                                 cart_p, "", reward, done])
                episode_id, step = episode_id + 1, 0
                continue

            u = policy(cart_v, pole_av, pole_a)
            writer.writerow([episode_id, step, cart_v, pole_av, pole_a,
                             cart_p, u, reward, done])
            step += 1
            sock.sendall(f"{u}\n".encode("ascii"))


if __name__ == "__main__":
    main()
