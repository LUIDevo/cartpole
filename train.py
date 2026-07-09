import numpy as np
import torch
import torch.nn as nn

# control the simulation with network
# after one episode, update the NN
# repeat 

# inputs:episode_id,step,cart_velocity,pole_angular_velocity,pole_angle,cart_position,motor_command,reward,done
# stripped inputs:cart_velocity,pole_angular_velocity,pole_angle,cart_position,reward
# weights are length 5
# define our output: motor command

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

def main():
    for iteration in range(10):
        for episode in range(100):
            while not done:
                with torch.no_grad():
                    action, log_prob= policy.sample(state)
                next_state, reward, done = sim.step(action)
                episode_log.append((state,action,reward))
                state=next_state
            batch.append(episode_log)

