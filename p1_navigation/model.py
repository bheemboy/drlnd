import torch
import torch.nn as nn


class QNetModel(nn.Module):
    """
    Dueling Q network
    """
    def __init__(self, state_size, action_size, seed, hidden_layers=None, dueling=True):
        """
        state_size (int): Number of parameters in the state
        action_size (int): Number of possible actions
        seed (int): Random seed
        hidden_layers []: Array with number of nodes in each hidden layer
        dueling (boolean): dueling network
        """
        super(QNetModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = dueling

        if hidden_layers is None:
            hidden_layers = [64, 64]
        hidden_layers_len = len(hidden_layers)

        if dueling and len(hidden_layers) < 2:
            raise Exception('Duelling DQN needs at least two hidden layers.')

        self.feature_layer = nn.Sequential()
        self.state_value_stream = None
        self.advantage_stream = None

        for i in range(hidden_layers_len):
            if i == 0:
                self.feature_layer.add_module(f'fc{i}', nn.Linear(state_size, hidden_layers[i]))
                self.feature_layer.add_module(f'ReLU{i}', nn.ReLU())
            else:
                if dueling and i == len(hidden_layers)-1:
                    # if dueling special handling of last layer
                    self.state_value_stream = nn.Sequential(nn.Linear(hidden_layers[i-1], hidden_layers[i]),
                                                            nn.ReLU(),
                                                            nn.Linear(hidden_layers[i], 1))
                    self.advantage_stream = nn.Sequential(nn.Linear(hidden_layers[i-1], hidden_layers[i]),
                                                          nn.ReLU(),
                                                          nn.Linear(hidden_layers[i], action_size))
                else:
                    self.feature_layer.add_module(f'fc{i}', nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                    self.feature_layer.add_module(f'ReLU{i}', nn.ReLU())
                    self.feature_layer.add_module('action', nn.Linear(hidden_layers[i], action_size))

    def forward(self, state):
        features = self.feature_layer(state)
        if self.dueling:
            state_values = self.state_value_stream(features)
            advantages = self.advantage_stream(features)
            return state_values + advantages - advantages.mean()
        else:
            return features

