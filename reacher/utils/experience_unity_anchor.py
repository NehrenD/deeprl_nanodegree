from unityagents import UnityEnvironment
import numpy as np
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import ptan
from utils.experience_unity import UnityExperienceSourceFirstLast

class Actor(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=300):

        super(Actor, self).__init__()
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


class TestAgent(ptan.agent.BaseAgent):

    def __init__(self, net, device="cpu", ou_enabled=True, ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2, ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_a_states


if __name__ == '__main__':
    env = UnityEnvironment(file_name='Reacher-2', no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    state_size = brain.vector_observation_space_size
    action_size = brain.vector_action_space_size

    model = Actor(state_size, action_size, 64, 64)
    agent = TestAgent(model)
    exp_source = UnityExperienceSourceFirstLast(env, agent, gamma=0.99, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=1000)


    for t in range(100):

        buffer.populate(10)

        rewards_steps = exp_source.pop_rewards_steps()
        if rewards_steps:
            rewards, steps = zip(*rewards_steps)
            print(rewards,steps)

        batch = buffer.sample(10)
        print(batch)