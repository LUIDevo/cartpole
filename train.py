import os

# This torch build has no kernels for the local GPU (sm_61) -> it would fall back
# to CPU anyway while spewing capability warnings. Hide the GPU before importing
# torch so the probe never runs. Training is CPU-bound on the sim regardless.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn

from sim_env import SimEnv

# PPO training loop:
#   collect a batch of episodes across NUM_ENVS parallel headless sims,
#   then run several clipped-surrogate epochs over the batch.
# PPO (vs plain REINFORCE): a learned value baseline slashes gradient variance,
# and the clipped ratio lets us safely take EPOCHS optimization passes per batch
# instead of one — much more learning per sim step.
#
# Rollout is vectorized: one Python thread steps ALL envs in lockstep with a
# single batched forward pass per tick (no per-env threads -> no GIL contention;
# send-all-then-recv-all overlaps the sims' physics computation).

# inputs: cart_velocity, pole_angular_velocity, pole_angle, cart_position
# output: motor command in [-1, 1]

GAMMA = 0.99          # reward discount
LR = 3e-4             # optimizer step size
ITERATIONS = 200      # training iterations (one batch + PPO update each)
NUM_ENVS = 24         # parallel sim processes (16C/32T box; leave headroom)
EPISODES = NUM_ENVS * 10  # episodes collected per update (split across the envs)
BASE_PORT = 9999      # envs listen on BASE_PORT, BASE_PORT+1, ...
MAX_STEPS = 1000      # per-episode cap (~16s at 60Hz); a balancing policy would
                      # otherwise never terminate and rollout would hang

EPOCHS = 10           # optimization passes over each collected batch
CLIP = 0.2            # PPO ratio clip
ENTROPY_COEF = 0.01   # exploration bonus (keeps std from collapsing too early)
VALUE_COEF = 0.5      # critic loss weight
INIT_STD = 0.5        # initial exploration noise; 1.0 drowned the policy mean
                      # (commands were near-random rail-to-rail)

class Network(nn.Module):
    """Actor-critic. Actor: Gaussian policy, tanh-squashed mean in [-1, 1] with a
    state-independent learnable log_std. Critic: state-value V(s) used as the
    advantage baseline (separate trunk; simple and stable at this scale)."""
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
        # tanh squashes the mean into the actuator range [-1, 1].
        return torch.tanh(self.actor(state)).squeeze(-1)

    def value(self, state):
        return self.critic(state).squeeze(-1)

    def dist(self, states):
        return torch.distributions.Normal(self(states), self.log_std.exp())

    @torch.no_grad()
    def act_batch(self, states):
        """Sample one action per env in a single forward pass (rollout only).

        states: [n, 4] tensor. Returns per-env lists: clamped motor commands sent
        to the sims, raw sampled actions, and their log-probs under the current
        (pre-update) policy — PPO's 'old' log-probs."""
        d = self.dist(states)
        actions = d.sample()                          # continuous, may exceed [-1,1]
        commands = actions.clamp(-1.0, 1.0)           # sim clamps too; keep aligned
        return commands.tolist(), actions.tolist(), d.log_prob(actions).tolist()

def discounted_returns(rewards, gamma=GAMMA):
    """Reward-to-go: R_t = r_t + gamma*r_{t+1} + ... (computed back to front)."""
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def collect(net, envs, per_env):
    """Vectorized rollout: step every env in lockstep from one thread.

    Each tick: stack the live envs' states, ONE batched forward to sample all
    actions, send every command (phase 1), then read every reply (phase 2) —
    the sims run their physics in parallel while Python does its bookkeeping.
    Each env contributes per_env episodes; finished envs go idle.

    Returns a list of episodes, each a list of (state, action, old_log_prob,
    reward)."""
    n = len(envs)
    batch = []
    episodes_left = [per_env] * n
    episode = [[] for _ in range(n)]
    steps = [0] * n
    states = [torch.tensor(env.reset(), dtype=torch.float32) for env in envs]
    active = list(range(n))

    while active:
        stacked = torch.stack([states[i] for i in active])
        commands, actions, old_lps = net.act_batch(stacked)

        for j, i in enumerate(active):
            envs[i].step_send(commands[j])

        still_active = []
        for j, i in enumerate(active):
            next_state, reward, done = envs[i].step_recv()
            episode[i].append((states[i], actions[j], old_lps[j], reward))
            steps[i] += 1

            if done or steps[i] >= MAX_STEPS:
                if not done:
                    envs[i].request_reset()  # step cap hit; force a fresh episode
                batch.append(episode[i])
                episode[i] = []
                steps[i] = 0
                episodes_left[i] -= 1
                if episodes_left[i] > 0:
                    states[i] = torch.tensor(envs[i].reset(), dtype=torch.float32)
                    still_active.append(i)
            else:
                states[i] = torch.tensor(next_state, dtype=torch.float32)
                still_active.append(i)
        active = still_active

    return batch


def update(net, optimizer, batch):
    """PPO update over a batch of episodes.

    batch: list of episodes, each a list of (state, action, old_log_prob, reward).
    Advantage = discounted return - V(s) (critic baseline), normalized.
    Runs EPOCHS passes of the clipped surrogate + value MSE + entropy bonus.
    Returns (last_loss, avg_episode_reward)."""
    states, actions, old_lps, returns = [], [], [], []
    total_reward = 0.0
    for episode in batch:
        rewards = [r for _, _, _, r in episode]
        total_reward += sum(rewards)
        returns.extend(discounted_returns(rewards))
        for s, a, lp, _ in episode:
            states.append(s)
            actions.append(a)
            old_lps.append(lp)

    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.float32)
    old_lps = torch.tensor(old_lps, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    # Advantages from the pre-update critic; fixed across the PPO epochs.
    with torch.no_grad():
        advantages = returns - net.value(states)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    loss = torch.tensor(0.0)
    for _ in range(EPOCHS):
        d = net.dist(states)
        new_lps = d.log_prob(actions)
        ratio = (new_lps - old_lps).exp()
        # Clipped surrogate: a ratio outside [1-CLIP, 1+CLIP] gets no extra credit,
        # so repeated epochs can't push the policy far from the data-collecting one.
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
    return loss.item(), total_reward / len(batch)


def make_envs():
    """Launch NUM_ENVS headless sim processes. Build once (first env), reuse the
    compiled output for the rest so we don't rebuild NUM_ENVS times."""
    envs = [SimEnv(port=BASE_PORT, build=True)]
    for i in range(1, NUM_ENVS):
        envs.append(SimEnv(port=BASE_PORT + i, build=False))
    return envs


def main():
    net = Network()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    envs = make_envs()
    per_env = EPISODES // NUM_ENVS
    try:
        for iteration in range(ITERATIONS):
            batch = collect(net, envs, per_env)
            avg_len = sum(len(ep) for ep in batch) / len(batch)
            loss, avg_reward = update(net, optimizer, batch)
            std = float(net.log_std.detach().exp())
            print(f"iter {iteration:3d}  avg_reward {avg_reward:8.2f}  "
                  f"avg_len {avg_len:6.1f}  std {std:5.3f}  loss {loss:8.4f}",
                  flush=True)
            if (iteration + 1) % 20 == 0:
                torch.save(net.state_dict(), "policy.pt")  # periodic checkpoint
    finally:
        for env in envs:
            env.close()

    torch.save(net.state_dict(), "policy.pt")
    print("saved policy.pt")


if __name__ == "__main__":
    main()
