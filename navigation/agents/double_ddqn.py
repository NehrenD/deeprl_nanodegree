import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from utils.replay_buffer import ReplayBuffer
from networks.dueling_dqn import DuelingDQNetwork
from utils.config import *


class DoubleDDQNAgent():
    "Implementes a Double Dueling DQN Agent"

    def __init__(self, state_size, action_size, seed, checkpoint=None):
        """
        Contructor

        :param state_size:
        :param action_size:
        :param seed:
        :param checkpoint: if running from a checkpoint
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(seed)

        # As for any DQN implementation  we create a local and a target Network.
        # In this Case we use the DuelingDQN Implementation for both networks

        self.qnetwork_local = DuelingDQNetwork(state_size, action_size, seed, fc1_units=FC1_UNITS,
                                               fc2_units=FC2_UNITS).to(device)
        self.qnetwork_target = DuelingDQNetwork(state_size, action_size, seed, fc1_units=FC1_UNITS,
                                                fc2_units=FC2_UNITS).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        if checkpoint:
            #If We have a checkpoint we load the state to the networks and optimizers
            print('Using Checkpoint...')
            self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
            self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Main Step function for the agent. Every UPDATE_EVERY time it runs a learning step
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions using an epsilong greedy approach.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if np.random.rand() > eps:
            return action_values.max(dim=1)[1].item()
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]):A batch of experiences
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # For DoubleQN use the local network to find the best action
        q_local_argmax = self.qnetwork_local(states).max(1)[1].unsqueeze(1)

        # GeEvaluate next state from target using best actions estimated from local
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, q_local_argmax)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
