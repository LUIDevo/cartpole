import numpy as np
import torch
import torch.nn as nn

from sim_env import SimEnv

# control the simulation with network
# after one episode, update the NN
# repeat

# inputs:episode_id,step,cart_velocity,pole_angular_velocity,pole_angle,cart_position,motor_command,reward,done
# stripped inputs:cart_velocity,pole_angular_velocity,pole_angle,cart_position,reward
# weights are length 5
# define our output: motor command

# The network picks a discrete action; map each to a motor command in [-1, 1].
ACTIONS = [-1.0, 1.0]  # bang-bang: full left / full right

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, state):
        return self.net(state)
    def sample(self, state):
        logits=self(state)
        dist=torch.distributions.Categorical(logits=logits)
        action=dist.sample()
        log_prob=dist.log_prob(action)
        return action, log_prob

def main():
    net = Network()
    with SimEnv(port=9999) as sim:
        for iteration in range(10):
            batch=[]
            for episode in range(100):
                done=False
                episode_log=[]
                state=torch.tensor(sim.reset(), dtype=torch.float32)
                while not done:
                    action, log_prob = net.sample(state)
                    command = ACTIONS[int(action)]           # discrete action -> motor command
                    next_state, reward, done = sim.step(command)
                    print(next_state, reward)
                    episode_log.append((log_prob, reward))
                    state=torch.tensor(next_state, dtype=torch.float32)
                batch.append(episode_log)
            


if __name__ == "__main__":
    main()
