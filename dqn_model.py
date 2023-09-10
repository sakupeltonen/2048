import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, max_val, board_height, board_width, n_actions):
        super(DQN, self).__init__()

        self.input_features = max_val * board_height * board_width

        # self.fc = nn.Sequential(
        #     nn.Linear(self.input_features, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_actions)
        # )
        self.fc = nn.Sequential(
            nn.Linear(self.input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        # no non-linearity on the last layer, because q values can be anything
        # TODO add more layers?

    def forward(self, x):
        input = x.view(x.shape[0], -1).float()  
        return self.fc(input)