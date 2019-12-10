import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        state_size (int): Number of parameters in the state
        action_size (int): Number of possible actions
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.q_value_stream = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

    def forward(self, state):
        """
        Maps state of the environment to actions
        """
        return self.q_value_stream(state)


class QNetDueling(nn.Module):
    """
    Dueling Q network
    """
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        state_size (int): Number of parameters in the state
        action_size (int): Number of possible actions
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetDueling, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.feauture_layer = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU()
        )

        self.state_value_stream = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        state_values = self.state_value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = state_values + advantages - advantages.mean()
        return qvals

