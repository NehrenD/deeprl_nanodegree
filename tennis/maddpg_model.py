import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
from collections import deque,namedtuple

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

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, full_state_size, full_action_size, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of both agents states
            action_size (int): Dimension of both agents actions
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(full_state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+full_action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, full_state, full_action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.elu(self.fcs1(full_state))
        x = torch.cat((xs, full_action), dim=1)
        x = F.elu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size,device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["full_state", "state", "action", "reward", \
                                                                "full_next_state", "next_state", "done"])
        self.device = device

    def add(self, full_state, state, action, reward, full_next_state, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(full_state, state, action, reward, full_next_state, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        full_states = torch.from_numpy(np.array([e.full_state for e in experiences if e is not None])).float().to(self.device)
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)
        full_next_states = torch.from_numpy(np.array([e.full_next_state for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (full_states, states, actions, rewards, full_next_states, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class MADDPGAgent(object):
    """
        This Agent is very similar to a DDPG Agent with the only difference that the critic network leverages the full states and actions for both agents
        The Actor only uses it's own network so that pst training there is no more sharing of information between the agents.
    
    """
        
    def __init__(self, state_size, action_size, num_agents,config):
        
        self.state_size = state_size
        self.action_size = action_size        
        self.config = config
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size,config.ACTOR_FC1_UNITS,config.ACTOR_FC2_UNITS).to(config.DEVICE)
        self.actor_target = Actor(state_size, action_size,config.ACTOR_FC1_UNITS,config.ACTOR_FC2_UNITS).to(config.DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.LR_ACTOR, weight_decay=config.WEIGHT_DECAY)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(num_agents*state_size, num_agents*action_size,config.CRITIC_FC1_UNITS,config.CRITIC_FC2_UNITS).to(config.DEVICE)
        self.critic_target = Critic(num_agents*state_size, num_agents*action_size,config.CRITIC_FC1_UNITS,config.CRITIC_FC2_UNITS).to(config.DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.LR_CRITIC, weight_decay=config.WEIGHT_DECAY)

        # Noise process        
        self.noise_scale = config.NOISE_START
    
        # Make sure target is initialized with the same weight as the source (makes a big difference)
        self.hard_update()

    def act(self, states, add_noise=True,noise_scale=1.0):
        """Returns actions for given state as per current policy."""
                
        if not add_noise:
            self.noise_scale = 0.0
                                    
        states = torch.from_numpy(states).float().to(self.config.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        
        #add noise
        actions += self.noise_scale*np.random.normal(0,0.5,self.action_size) #works much better than OU Noise process
        
        return np.clip(actions, -1, 1)
        
    def learning_step(self, experiences, gamma):
        """
        Leraning 
        """
        
        full_states, actor_full_actions, full_actions, agent_rewards, agent_dones, full_next_states, critic_full_next_actions = experiences
        
        # Learning Step for Critic using full state, action and next state and full next action#

        #Q Target
        Q_target_next = self.critic_target(full_next_states, critic_full_next_actions)        
        Q_target = agent_rewards + gamma * Q_target_next * (1 - agent_dones)
        
        #Q Expected
        Q_expected = self.critic_local(full_states, full_actions)
        
        #Loss
        critic_loss = F.mse_loss(input=Q_expected, target=Q_target)
        
        #Training Step
        self.critic_optimizer.zero_grad()
        critic_loss.backward()        
        self.critic_optimizer.step()
        
        # Learning Step for Actor #
        
        #Loss
        actor_loss = -self.critic_local.forward(full_states, actor_full_actions).mean() 
        
        #Training Step
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()                  

        return (critic_loss,actor_loss)
                
    def soft_update(self):
        """
        Soft update traget networks for actor an critic
        """
        self._soft_update(self.critic_local, self.critic_target, self.config.TAU)
        self._soft_update(self.actor_local, self.actor_target, self.config.TAU)
   
    def _soft_update(self, local_model, target_model, tau):
        """
        Soft Update
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self):
        """
        Soft update traget networks for actor an critic
        """
        self._hard_update(self.critic_local, self.critic_target)
        self._hard_update(self.actor_local, self.actor_target)
        
    def _hard_update(self, target, source):
        """
        Hard Update to be used at initialization
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)