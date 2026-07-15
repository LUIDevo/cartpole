import argparse
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

from math_env import MathCartPoleVec, STATE_DIM

torch.set_num_threads(4)

GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
ITERATIONS = 3000
NUM_ENVS = 64
STEPS_PER_ITER = 1024 * NUM_ENVS
MAX_STEPS = 3000

EPOCHS = 4
MINIBATCHES = 4
TARGET_KL = 0.02
CLIP = 0.2
ENTROPY_COEF = 1e-3
VALUE_COEF = 0.5
INIT_STD = 0.5
MIN_STD = 0.10
MAX_STD = 0.6


class Network(nn.Module):
    def __init__(self, init_std=INIT_STD):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.full((1,), float(np.log(init_std))))

    def forward(self, state):
        return torch.tanh(self.actor(state)).squeeze(-1)

    def value(self, state):
        return self.critic(state).squeeze(-1)

    def dist(self, states):
        return torch.distributions.Normal(self(states), self.log_std.exp())

    @torch.no_grad()
    def act_batch(self, states):
        mean = self(states)
        std = self.log_std.exp()
        actions = mean + std * torch.randn_like(mean)
        log_probs = (-0.5 * ((actions - mean) / std) ** 2
                     - self.log_std - 0.5 * math.log(2 * math.pi))
        return actions.clamp(-1.0, 1.0), actions, log_probs


class VecRunner:
    def __init__(self, env, max_steps=MAX_STEPS):
        self.env = env
        self.max_steps = max_steps
        self.obs = torch.from_numpy(env.reset_all())
        self.ep_len = np.zeros(env.n, dtype=np.int64)
        self.ep_rew = np.zeros(env.n)

    def collect(self, net, budget):
        n = self.env.n
        T = budget // n
        states = torch.empty((T, n, STATE_DIM))
        actions = torch.empty((T, n))
        old_lps = torch.empty((T, n))
        rewards = torch.empty((T, n))
        terminal = torch.zeros((T, n), dtype=torch.bool)
        truncated = torch.zeros((T, n), dtype=torch.bool)
        boot_idx, boot_obs = [], []
        ep_rewards, ep_lens = [], []

        for t in range(T):
            commands, acts, lps = net.act_batch(self.obs)
            obs, rew, done = self.env.step(commands.numpy())

            states[t] = self.obs
            actions[t] = acts
            old_lps[t] = lps
            rewards[t] = torch.from_numpy(rew).float()

            self.ep_len += 1
            self.ep_rew += rew
            trunc = ~done & (self.ep_len >= self.max_steps)
            terminal[t] = torch.from_numpy(done)
            truncated[t] = torch.from_numpy(trunc)

            reset_mask = done | trunc
            if reset_mask.any():
                for i in np.flatnonzero(trunc):
                    boot_idx.append((t, i))
                    boot_obs.append(obs[i])
                for i in np.flatnonzero(reset_mask):
                    ep_lens.append(int(self.ep_len[i]))
                    ep_rewards.append(float(self.ep_rew[i]))
                self.ep_len[reset_mask] = 0
                self.ep_rew[reset_mask] = 0.0
                obs = self.env.reset_where(reset_mask)
            self.obs = torch.from_numpy(obs)

        with torch.no_grad():
            values = net.value(states.reshape(T * n, STATE_DIM)).reshape(T, n)
            next_values = torch.empty((T, n))
            next_values[:-1] = values[1:]
            next_values[-1] = net.value(self.obs)
            next_values[terminal] = 0.0
            if boot_obs:
                boots = net.value(torch.from_numpy(np.stack(boot_obs)))
                for k, (t, i) in enumerate(boot_idx):
                    next_values[t, i] = boots[k]

        not_done = (~(terminal | truncated)).float()
        advantages = torch.empty((T, n))
        last_gae = torch.zeros(n)
        for t in reversed(range(T)):
            delta = rewards[t] + GAMMA * next_values[t] - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * not_done[t] * last_gae
            advantages[t] = last_gae
        returns = advantages + values

        data = (states.reshape(-1, STATE_DIM), actions.reshape(-1), old_lps.reshape(-1),
                advantages.reshape(-1), returns.reshape(-1))
        return data, ep_rewards, ep_lens, goal_scores(states)


GOAL_NAMES = ("uu", "ud", "du", "dd")


def goal_scores(states):
    g1, g2 = states[..., 6], states[..., 7]
    c1 = torch.cos(states[..., 2] * math.pi)
    c2 = torch.cos(states[..., 5] * math.pi)
    score = 0.5 * (g1 * c1 + g2 * c2)
    masks = {"uu": (g1 > 0) & (g2 > 0), "ud": (g1 > 0) & (g2 < 0),
             "du": (g1 < 0) & (g2 > 0), "dd": (g1 < 0) & (g2 < 0)}
    return {k: float(score[m].mean()) if m.any() else float("nan")
            for k, m in masks.items()}


def update(net, optimizer, states, actions, old_lps, advantages, returns,
           min_std=MIN_STD, max_std=MAX_STD):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch = states.shape[0]
    mb_size = batch // MINIBATCHES
    loss = torch.tensor(0.0)
    for _ in range(EPOCHS):
        perm = torch.randperm(batch)
        stop = False
        for k in range(MINIBATCHES):
            mb = perm[k * mb_size:(k + 1) * mb_size]
            d = net.dist(states[mb])
            new_lps = d.log_prob(actions[mb])
            ratio = (new_lps - old_lps[mb]).exp()
            surrogate = torch.min(
                ratio * advantages[mb],
                ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * advantages[mb],
            ).mean()
            value_loss = (net.value(states[mb]) - returns[mb]).pow(2).mean()
            entropy = d.entropy().mean()

            loss = -surrogate + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            net.log_std.data.clamp_(np.log(min_std), np.log(max_std))

            with torch.no_grad():
                approx_kl = (old_lps[mb] - new_lps).mean()
            if approx_kl > TARGET_KL:
                stop = True
                break
        if stop:
            break
    return loss.item()


GOAL_VECS = {"uu": (1.0, 1.0), "ud": (1.0, -1.0), "du": (-1.0, 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", choices=sorted(GOAL_VECS),
                    help="train a single-goal specialist (dd needs no policy)")
    ap.add_argument("--init", help="checkpoint to warm-start from")
    ap.add_argument("--out", help="output checkpoint path")
    ap.add_argument("--iters", type=int, default=ITERATIONS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--log", help="training log csv path")
    ap.add_argument("--balance-frac", type=float, default=None,
                    help="fraction of resets inside the up-up capture basin")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS,
                    help="episode length before truncation")
    ap.add_argument("--std-init", type=float, default=INIT_STD)
    ap.add_argument("--std-min", type=float, default=MIN_STD)
    ap.add_argument("--critic-warmup", type=int, default=0,
                    help="critic-only iterations before PPO updates "
                         "(use when warm-starting from a distilled policy "
                         "whose critic is untrained)")
    ap.add_argument("--handoff", action="store_true",
                    help="end episodes with a bonus once both poles enter "
                         "the balance specialist's catch basin (train a "
                         "swing specialist)")
    args = ap.parse_args()

    out = args.out or (f"policy_{args.goal}.pt" if args.goal else "policy.pt")
    fixed = GOAL_VECS[args.goal] if args.goal else None
    log_path = args.log or (f"training_log_{args.goal}.csv" if args.goal
                            else "training_log.csv")

    net = Network(args.std_init)
    if args.init:
        net.load_state_dict(torch.load(args.init, weights_only=True))
        net.log_std.data.fill_(float(np.log(args.std_init)))
        print(f"warm-started from {args.init}")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    env_kw = {} if args.balance_frac is None else \
        {"balance_frac": args.balance_frac}
    if args.handoff:
        env_kw["handoff"] = True
    runner = VecRunner(MathCartPoleVec(NUM_ENVS, fixed_goal=fixed, **env_kw),
                       max_steps=args.max_steps)
    log = open(log_path, "w")
    log.write("iter,avg_reward,avg_len,episodes,std,loss,"
              "score_uu,score_ud,score_du,score_dd\n")
    try:
        run(net, optimizer, runner, log, args.iters, out,
            lr=args.lr, std_min=args.std_min,
            critic_warmup=args.critic_warmup)
    except KeyboardInterrupt:
        print(f"\ninterrupted — saving {out}")
    log.close()
    torch.save(net.state_dict(), out)
    print(f"saved {out}")

    if args.goal == "uu" and out == "policy_uu.pt":
        for other in ("ud", "du"):
            path = f"policy_{other}.pt"
            if not os.path.exists(path):
                torch.save(net.state_dict(), path)
                print(f"seeded {path} from uu weights "
                      f"(fine-tune with: py train.py --goal {other} --init {path})")


def run(net, optimizer, runner, log, iters, out,
        lr=LR, std_min=MIN_STD, critic_warmup=0):
    if critic_warmup:
        critic_opt = torch.optim.Adam(net.critic.parameters(), lr=1e-3)
        for i in range(critic_warmup):
            data, *_ = runner.collect(net, STEPS_PER_ITER)
            states, _, _, _, returns = data
            for _ in range(EPOCHS):
                vloss = (net.value(states) - returns).pow(2).mean()
                critic_opt.zero_grad()
                vloss.backward()
                critic_opt.step()
            print(f"critic warmup {i}  vloss {vloss.item():8.2f}", flush=True)
    for iteration in range(iters):
        frac = max(0.1, 1.0 - iteration / iters)
        for group in optimizer.param_groups:
            group["lr"] = lr * frac
        data, ep_rewards, ep_lens, scores = runner.collect(net, STEPS_PER_ITER)
        loss = update(net, optimizer, *data, min_std=std_min)
        std = float(net.log_std.detach().exp())
        score_str = "  ".join(f"{k} {scores[k]:+.2f}" for k in GOAL_NAMES)
        if ep_lens:
            stats = (f"avg_reward {sum(ep_rewards)/len(ep_rewards):8.2f}  "
                     f"eps {len(ep_lens):3d}")
        else:
            stats = "avg_reward      n/a  eps   0"
        print(f"iter {iteration:3d}  {stats}  {score_str}  std {std:5.3f}  "
              f"loss {loss:8.4f}", flush=True)
        score_csv = ",".join(f"{scores[k]:.4f}" for k in GOAL_NAMES)
        if ep_lens:
            log.write(f"{iteration},{sum(ep_rewards)/len(ep_rewards):.4f},"
                      f"{sum(ep_lens)/len(ep_lens):.2f},{len(ep_lens)},"
                      f"{std:.4f},{loss:.4f},{score_csv}\n")
        else:
            log.write(f"{iteration},,,0,{std:.4f},{loss:.4f},{score_csv}\n")
        log.flush()
        if (iteration + 1) % 20 == 0:
            torch.save(net.state_dict(), out)


if __name__ == "__main__":
    main()
