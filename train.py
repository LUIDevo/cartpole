import os
import threading

# This torch build has no kernels for the local GPU (sm_61) -> it would fall back
# to CPU anyway while spewing capability warnings. Hide the GPU before importing
# torch so the probe never runs. Training is CPU-bound on the sim regardless.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn

from sim_env import SimEnv

# control the simulation with network
# after one batch of episodes, update the NN
# repeat
#
# Training runs headless (no window) and across NUM_ENVS parallel sim processes,
# so many episodes are collected at once instead of one-at-a-time.

# inputs:episode_id,step,cart_velocity,pole_angular_velocity,pole_angle,cart_position,motor_command,reward,done
# stripped inputs:cart_velocity,pole_angular_velocity,pole_angle,cart_position,reward
# weights are length 5
# define our output: motor command

GAMMA = 0.99          # reward discount
LR = 1e-3             # optimizer step size
ITERATIONS = 200      # policy updates (REINFORCE needs many small steps)
EPISODES = 96         # episodes collected per update (split across the envs)
NUM_ENVS = 8          # parallel sim processes
BASE_PORT = 9999      # envs listen on BASE_PORT, BASE_PORT+1, ...
MAX_STEPS = 1000      # per-episode cap (~16s at 60Hz); a balancing policy would
                      # otherwise never terminate and rollout would hang

class Network(nn.Module):
    """Continuous Gaussian policy: outputs the mean motor command in [-1, 1];
    a state-independent learnable log_std sets exploration noise."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # log std-dev, shared across states. exp(0)=1 -> broad initial exploration.
        self.log_std = nn.Parameter(torch.zeros(1))
    def forward(self, state):
        # tanh squashes the mean into the actuator range [-1, 1].
        return torch.tanh(self.net(state)).squeeze(-1)
    @torch.no_grad()
    def act(self, state):
        """Sample one action for env rollout. no_grad -> safe to call from many
        collection threads at once (builds no autograd graph). Returns the clamped
        motor command sent to the sim and the raw sampled action (stored for the
        later log-prob recompute)."""
        mean = self(state)
        std = self.log_std.exp()
        action = torch.normal(mean, std)              # continuous, may exceed [-1,1]
        command = float(action.clamp(-1.0, 1.0))      # sim clamps too; keep aligned
        return command, float(action)
    def log_prob(self, states, actions):
        """Log-prob of taken actions under the current policy (batched, with grad)."""
        mean = self(states)
        std = self.log_std.exp()
        return torch.distributions.Normal(mean, std).log_prob(actions)

def discounted_returns(rewards, gamma=GAMMA):
    """Reward-to-go: R_t = r_t + gamma*r_{t+1} + ... (computed back to front)."""
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def run_episodes(net, sim, n_episodes, out):
    """Roll out n_episodes on one sim, appending each to `out` (thread's own list).

    Each episode is a list of (state, action, reward). log_prob is NOT computed
    here — it is recomputed in one batched pass at update time (same params ->
    identical value) so no autograd graph is built across threads."""
    for _ in range(n_episodes):
        episode = []
        state = torch.tensor(sim.reset(), dtype=torch.float32)
        done = False
        for _ in range(MAX_STEPS):
            command, action = net.act(state)
            next_state, reward, done = sim.step(command)
            episode.append((state, action, reward))
            state = torch.tensor(next_state, dtype=torch.float32)
            if done:
                break
        if not done:
            sim.request_reset()  # step cap hit; force the sim to start a new episode
        out.append(episode)


def update(net, optimizer, batch):
    """REINFORCE step over a batch of episodes.

    batch: list of episodes, each a list of (state, action, reward) tuples.
    Loss = -mean over all steps of  log_prob * advantage,
    where advantage = discounted return, baselined by the batch mean and
    scaled to unit variance to keep the gradient well-conditioned.
    Returns (loss, avg_episode_reward)."""
    states, actions, returns = [], [], []
    total_reward = 0.0
    for episode in batch:
        rewards = [r for _, _, r in episode]
        total_reward += sum(rewards)
        returns.extend(discounted_returns(rewards))
        states.extend(s for s, _, _ in episode)
        actions.extend(a for _, a, _ in episode)

    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    # baseline + normalize (variance reduction)
    advantages = returns - returns.mean()
    advantages = advantages / (advantages.std() + 1e-8)

    log_probs = net.log_prob(states, actions)
    loss = -(log_probs * advantages).mean()

    optimizer.zero_grad()
    loss.backward()
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
            outs = [[] for _ in envs]
            threads = [
                threading.Thread(target=run_episodes, args=(net, env, per_env, out))
                for env, out in zip(envs, outs)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            batch = [episode for out in outs for episode in out]
            loss, avg_reward = update(net, optimizer, batch)
            print(f"iter {iteration:3d}  avg_reward {avg_reward:8.2f}  loss {loss:8.4f}")
    finally:
        for env in envs:
            env.close()

    torch.save(net.state_dict(), "policy.pt")
    print("saved policy.pt")


if __name__ == "__main__":
    main()
