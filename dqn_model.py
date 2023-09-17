import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, max_val, board_height, board_width, layer_size, n_actions):
        super(DQN, self).__init__()

        self.input_features = max_val * board_height * board_width

        self.fc = nn.Sequential(
            # nn.Linear(self.input_features, layer_size),
            # nn.ReLU(),
            # nn.Linear(layer_size, layer_size),
            # nn.ReLU(),
            # nn.Linear(layer_size, layer_size),
            # nn.ReLU(),
            # nn.Linear(layer_size, layer_size),
            # nn.ReLU(),
            # nn.Linear(layer_size, n_actions)
            nn.Linear(self.input_features, n_actions, bias=False)
        )
        # no non-linearity on the last layer, because q values can be anything
        # TODO add more layers?

    def forward(self, x):
        input = x.view(x.shape[0], -1).float()  
        return self.fc(input)