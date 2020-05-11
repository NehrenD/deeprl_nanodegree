import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        Constructor. Creates a Dueling DQN Network
        The first 2 layers are shared
        The third layer has 2 heads:
        -One for the value function estimation
        -The second one for the advantages estimation

        :param state_size:
        :param action_size:
        :param seed:
        :param fc1_units:
        :param fc2_units:
        """
        super(DuelingDQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        self.fc3_value = nn.Linear(fc2_units, 1)

        self.fc3_advantage = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        Use the Value and Advantages to create the estimate onf the QValue

        :param state:
        :return: The estimated Q Value
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        value = self.fc3_value(x)
        advantages = self.fc3_advantage(x)

        q_values = value + (advantages - advantages.mean())

        return q_values
