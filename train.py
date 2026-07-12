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
ENTROPY_COEF = 0.0
VALUE_COEF = 0.5
INIT_STD = 0.5
MIN_STD = 0.05
MAX_STD = 0.6


class Network(nn.Module):
    def __init__(self):
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
        self.log_std = nn.Parameter(torch.full((1,), float(torch.log(torch.tensor(INIT_STD)))))

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
    def __init__(self, env):
        self.env = env
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
            trunc = ~done & (self.ep_len >= MAX_STEPS)
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
        return data, ep_rewards, ep_lens


def update(net, optimizer, states, actions, old_lps, advantages, returns):
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
            net.log_std.data.clamp_(np.log(MIN_STD), np.log(MAX_STD))

            with torch.no_grad():
                approx_kl = (old_lps[mb] - new_lps).mean()
            if approx_kl > TARGET_KL:
                stop = True
                break
        if stop:
            break
    return loss.item()


def main():
    net = Network()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    runner = VecRunner(MathCartPoleVec(NUM_ENVS))
    log = open("training_log.csv", "w")
    log.write("iter,avg_reward,avg_len,episodes,std,loss\n")
    for iteration in range(ITERATIONS):
        frac = max(0.1, 1.0 - iteration / ITERATIONS)
        for group in optimizer.param_groups:
            group["lr"] = LR * frac
        data, ep_rewards, ep_lens = runner.collect(net, STEPS_PER_ITER)
        loss = update(net, optimizer, *data)
        std = float(net.log_std.detach().exp())
        if ep_lens:
            stats = (f"avg_reward {sum(ep_rewards)/len(ep_rewards):8.2f}  "
                     f"avg_len {sum(ep_lens)/len(ep_lens):6.1f}  "
                     f"eps {len(ep_lens):3d}")
        else:
            stats = "avg_reward      n/a  avg_len    n/a  eps   0"
        print(f"iter {iteration:3d}  {stats}  std {std:5.3f}  loss {loss:8.4f}",
              flush=True)
        if ep_lens:
            log.write(f"{iteration},{sum(ep_rewards)/len(ep_rewards):.4f},"
                      f"{sum(ep_lens)/len(ep_lens):.2f},{len(ep_lens)},"
                      f"{std:.4f},{loss:.4f}\n")
        else:
            log.write(f"{iteration},,,0,{std:.4f},{loss:.4f}\n")
        log.flush()
        if (iteration + 1) % 20 == 0:
            torch.save(net.state_dict(), "policy.pt")

    log.close()
    torch.save(net.state_dict(), "policy.pt")
    print("saved policy.pt")


if __name__ == "__main__":
    main()
