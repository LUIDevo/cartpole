import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn

from sim_env import SimEnv

GAMMA = 0.99
LR = 3e-4
ITERATIONS = 100
NUM_ENVS = 24
STEPS_PER_ITER = 1024 * NUM_ENVS
BASE_PORT = 9999
MAX_STEPS = 1000

EPOCHS = 10
CLIP = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
INIT_STD = 0.5


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
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
        d = self.dist(states)
        actions = d.sample()
        commands = actions.clamp(-1.0, 1.0)
        return commands.tolist(), actions.tolist(), d.log_prob(actions).tolist()


def discounted_returns(rewards, bootstrap=0.0, gamma=GAMMA):
    returns = []
    running = float(bootstrap)
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


class VecRunner:
    def __init__(self, envs):
        self.envs = envs
        self.states = [torch.tensor(env.reset(), dtype=torch.float32) for env in envs]
        self.steps = [0] * len(envs)

    def collect(self, net, budget):
        n = len(self.envs)
        episode = [[] for _ in range(n)]
        all_states, all_actions, all_lps, all_returns = [], [], [], []
        ep_rewards, ep_lens = [], []

        def finalize(i, bootstrap):
            states_i, actions_i, lps_i, rewards_i = zip(*episode[i])
            all_states.extend(states_i)
            all_actions.extend(actions_i)
            all_lps.extend(lps_i)
            all_returns.extend(discounted_returns(rewards_i, bootstrap))
            episode[i] = []

        collected = 0
        while collected < budget:
            stacked = torch.stack(self.states)
            commands, actions, old_lps = net.act_batch(stacked)

            for i, env in enumerate(self.envs):
                env.step_send(commands[i])

            for i, env in enumerate(self.envs):
                next_state, reward, done = env.step_recv()
                episode[i].append((self.states[i], actions[i], old_lps[i], reward))
                self.steps[i] += 1
                collected += 1

                if done:
                    ep_lens.append(self.steps[i])
                    ep_rewards.append(sum(r for _, _, _, r in episode[i]))
                    finalize(i, bootstrap=0.0)
                    self.steps[i] = 0
                    self.states[i] = torch.tensor(env.reset(), dtype=torch.float32)
                elif self.steps[i] >= MAX_STEPS:
                    env.request_reset()
                    ep_lens.append(self.steps[i])
                    ep_rewards.append(sum(r for _, _, _, r in episode[i]))
                    next_t = torch.tensor(next_state, dtype=torch.float32)
                    with torch.no_grad():
                        finalize(i, bootstrap=net.value(next_t))
                    self.steps[i] = 0
                    self.states[i] = torch.tensor(env.reset(), dtype=torch.float32)
                else:
                    self.states[i] = torch.tensor(next_state, dtype=torch.float32)

        pending = [i for i in range(n) if episode[i]]
        if pending:
            with torch.no_grad():
                values = net.value(torch.stack([self.states[i] for i in pending]))
            for j, i in enumerate(pending):
                finalize(i, bootstrap=float(values[j]))

        data = (
            torch.stack(all_states),
            torch.tensor(all_actions, dtype=torch.float32),
            torch.tensor(all_lps, dtype=torch.float32),
            torch.tensor(all_returns, dtype=torch.float32),
        )
        return data, ep_rewards, ep_lens


def update(net, optimizer, states, actions, old_lps, returns):
    with torch.no_grad():
        advantages = returns - net.value(states)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    loss = torch.tensor(0.0)
    for _ in range(EPOCHS):
        d = net.dist(states)
        new_lps = d.log_prob(actions)
        ratio = (new_lps - old_lps).exp()
        surrogate = torch.min(
            ratio * advantages,
            ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * advantages,
        ).mean()
        value_loss = (net.value(states) - returns).pow(2).mean()
        entropy = d.entropy().mean()

        loss = -surrogate + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()
    return loss.item()


def make_envs():
    envs = [SimEnv(port=BASE_PORT, build=True)]
    for i in range(1, NUM_ENVS):
        envs.append(SimEnv(port=BASE_PORT + i, build=False))
    return envs


def main():
    net = Network()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    envs = make_envs()
    try:
        runner = VecRunner(envs)
        for iteration in range(ITERATIONS):
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
            if (iteration + 1) % 20 == 0:
                torch.save(net.state_dict(), "policy.pt")
    finally:
        for env in envs:
            env.close()

    torch.save(net.state_dict(), "policy.pt")
    print("saved policy.pt")


if __name__ == "__main__":
    main()
