import torch
import torch.nn as nn
import numpy as np
import ptan

#UTILS
class Config():

    #NETWORK

    ACTOR_FC1_UNITS = 128
    ACTOR_FC2_UNITS = 64
    CRITIC_FC1_UNITS = 128
    CRITIC_FC2_UNITS = 64
    NOISE_THETA = 0.15
    NOISE_SIGMA = 0.2
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-4
    TAU = 1e-4

    #REPLAY BUFFER
    BUFFER_SIZE = int(1e6)
    BATCH_SIZE = 256

    GAMMA = 0.99

    device = 'cpu'


class AgentDDPG(ptan.agent.BaseAgent):

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