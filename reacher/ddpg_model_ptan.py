import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import random
from collections import deque,namedtuple

#UTILS
class Config():

    #NETWORK

    ACTOR_FC1_UNITS = 64
    ACTOR_FC2_UNITS = 64
    CRITIC_FC1_UNITS = 64
    CRITIC_FC2_UNITS = 64
    NOISE_THETA = 0.15
    NOISE_SIGMA = 0.2
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    TAU = 1e-3
    WEIGHT_DECAY = 0  # L2 weight decay

    #REPLAY BUFFER
    BUFFER_SIZE = int(1e6)
    BATCH_SIZE = 128

    GAMMA = 0.99

    #RUNS
    MAX_STEPS = 500
    TRAIN_STEPS = 20
    TRAIN_TIMES = 10

    device = 'cpu'


class DDPGActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300):
        super(DDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class DDPGCritic(nn.Module):
    def __init__(self, state_size, action_size,fc1_units=400, fc2_units=300):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(fc1_units + action_size, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))