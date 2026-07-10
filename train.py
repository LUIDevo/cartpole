import os

# This torch build has no kernels for the local GPU (sm_61) -> it would fall back
# to CPU anyway while spewing capability warnings. Hide the GPU before importing
# torch so the probe never runs. Training is CPU-bound on the sim regardless.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn

from sim_env import SimEnv

# PPO training loop:
#   collect a fixed budget of steps across NUM_ENVS parallel headless sims,
#   then run several clipped-surrogate epochs over the batch.
# PPO (vs plain REINFORCE): a learned value baseline slashes gradient variance,
# and the clipped ratio lets us safely take EPOCHS optimization passes per batch
# instead of one — much more learning per sim step.
#
# Rollout is vectorized: one Python thread steps ALL envs in lockstep with a
# single batched forward pass per tick (no per-env threads -> no GIL contention;
# send-all-then-recv-all overlaps the sims' physics computation).
#
# Each iteration collects STEPS_PER_ITER steps, NOT a fixed number of episodes:
# with an episode quota, iteration time balloons as the policy learns to survive
# longer. Episodes cut off by the budget are bootstrapped with the critic
# (return of the unfinished tail ~ V(next_state)) and simply continue into the
# next iteration — constant wall time per iteration, no discarded sim work.

# inputs: cart_velocity, pole_angular_velocity, pole_angle, cart_position
# output: motor command in [-1, 1]

GAMMA = 0.99          # reward discount
LR = 3e-4             # optimizer step size
ITERATIONS = 100      # training iterations (one batch + PPO update each)
NUM_ENVS = 24         # parallel sim processes (16C/32T box; leave headroom)
STEPS_PER_ITER = 1024 * NUM_ENVS  # fixed data budget per iteration
BASE_PORT = 9999      # envs listen on BASE_PORT, BASE_PORT+1, ...
MAX_STEPS = 1000      # per-episode cap (~16s at 60Hz); rerandomizes the physics
                      # even if the policy can balance indefinitely

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


def discounted_returns(rewards, bootstrap=0.0, gamma=GAMMA):
    """Reward-to-go: R_t = r_t + gamma*r_{t+1} + ... (computed back to front).

    bootstrap: value of the state AFTER the last step — 0 for terminal episodes,
    V(next_state) for episodes truncated by the step budget or MAX_STEPS."""
    returns = []
    running = float(bootstrap)
    for r in reversed(rewards):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


class VecRunner:
    """Steps NUM_ENVS sims in lockstep and slices the experience stream into
    fixed-size PPO batches. Env state persists across collect() calls, so an
    episode interrupted by the step budget continues in the next iteration."""

    def __init__(self, envs):
        self.envs = envs
        self.states = [torch.tensor(env.reset(), dtype=torch.float32) for env in envs]
        self.steps = [0] * len(envs)   # steps taken in each env's current episode

    def collect(self, net, budget):
        """Collect >= budget total steps. Returns (states, actions, old_log_probs,
        returns) tensors plus (episode_rewards, episode_lengths) for episodes that
        FINISHED during this call (terminal or MAX_STEPS)."""
        n = len(self.envs)
        episode = [[] for _ in range(n)]   # this call's transitions per env
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
                    # Episode stats cover its full length, which may span calls.
                    ep_lens.append(self.steps[i])
                    ep_rewards.append(sum(r for _, _, _, r in episode[i]))
                    finalize(i, bootstrap=0.0)          # true terminal: no value tail
                    self.steps[i] = 0
                    self.states[i] = torch.tensor(env.reset(), dtype=torch.float32)
                elif self.steps[i] >= MAX_STEPS:
                    env.request_reset()                 # rerandomize the physics
                    ep_lens.append(self.steps[i])
                    ep_rewards.append(sum(r for _, _, _, r in episode[i]))
                    next_t = torch.tensor(next_state, dtype=torch.float32)
                    with torch.no_grad():
                        finalize(i, bootstrap=net.value(next_t))  # alive: bootstrap
                    self.steps[i] = 0
                    self.states[i] = torch.tensor(env.reset(), dtype=torch.float32)
                else:
                    self.states[i] = torch.tensor(next_state, dtype=torch.float32)

        # Budget hit: bootstrap every unfinished episode with V(current state) and
        # let it continue into the next collect() call.
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
    """PPO update: EPOCHS passes of clipped surrogate + value MSE + entropy bonus.
    Advantage = return - V(s) from the pre-update critic, normalized."""
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
    return loss.item()


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
    runner = VecRunner(envs)
    try:
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
                torch.save(net.state_dict(), "policy.pt")  # periodic checkpoint
    finally:
        for env in envs:
            env.close()

    torch.save(net.state_dict(), "policy.pt")
    print("saved policy.pt")


if __name__ == "__main__":
    main()
