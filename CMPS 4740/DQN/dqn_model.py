import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1) # Output single Q-value for the (state, action) pair

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # We don't store 'action' separately because our features INCLUDE the action
        # So 'state' here is actually phi(s, a)
        self.buffer.append((state, reward, next_state, done))

    def sample(self, batch_size):
        state, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)