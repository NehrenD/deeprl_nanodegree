
import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.replay_buffer import ReplayBuffer
from utils.ou_noise import OUNoise

from networks.ddpg_actor import DDPGActor
from networks.ddpg_critic import DDPGCritic


from utils.config import *

class DDPGAgent():

    def __init__(self ,state_size, action_size, random_seed,config):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Networks
        self.actor_local = DDPGActor(state_size, action_size, random_seed ,config.ACTOR_FC1_UNITS ,config.ACTOR_FC2_UNITS).to(config.device)
        self.actor_target = DDPGActor(state_size, action_size, random_seed ,config.ACTOR_FC1_UNITS ,config.ACTOR_FC2_UNITS).to(config.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.LR_ACTOR)

        # Critic Networks
        self.critic_local = DDPGCritic(state_size, action_size, random_seed ,config.CRITIC_FC1_UNITS ,config.CRITIC_FC2_UNITS).to \
            (config.device)
        self.critic_target = DDPGCritic(state_size, action_size, random_seed ,config.CRITIC_FC1_UNITS ,config.CRITIC_FC2_UNITS).to \
            (config.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.LR_CRITIC, weight_decay=config.WEIGHT_DECAY)

        self.noise = OUNoise(action_size ,random_seed ,config.NOISE_THETA ,config.NOISE_SIGMA)

        self.memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, random_seed)
        self.config = config

    def add_to_memory(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)

    def learning_step(self):
        if len(self.memory) > self.config.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.config.GAMMA)

    def act(self, state, add_noise=True):

        state = torch.from_numpy(state).float().to(self.config.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions.float())
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.config.TAU)

    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau *local_param.data + (1.0 -tau ) *target_param.data)
