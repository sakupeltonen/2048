import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, specs):
        super(DQN, self).__init__()
        n_actions = 4  # hardcoded for now
        n_extra_features = 12  # hardcoded for now

        self.input_features = specs['max_val'] * specs['height'] * specs['width'] + n_extra_features

        self.fc = nn.Sequential(
            nn.Linear(self.input_features, specs['layer_size']),
            nn.ReLU(),
            nn.Linear(specs['layer_size'], specs['layer_size']),
            nn.ReLU(),
            nn.Linear(specs['layer_size'], specs['layer_size']),
            nn.ReLU(),
            nn.Linear(specs['layer_size'], specs['layer_size']),
            nn.ReLU(),
            nn.Linear(specs['layer_size'], n_actions)
            # no non-linearity on the last layer, because q values can be anything
        )

    def forward(self, x):
        input = x.view(x.shape[0], -1).float()  
        return self.fc(input)
    
    @classmethod
    def from_file(cls, path, device, specs):
        model = cls(specs)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        return model
    

class DQNConv(nn.Module):
    def __init__(self, specs):
        super(DQNConv, self).__init__()
        n_actions = 4  # hardcoded for now
        n_extra_features = 12  # hardcoded for now

        self.specs = specs

        # 2D Convolutional layer for board
        self.conv = nn.Conv2d(in_channels=specs['max_val'], out_channels=32, kernel_size=2, stride=1)

        # Calculate the output dimensions of the convolutional layer
        conv_out_height = specs['height'] - 1  # 2x2 kernel with stride 1
        conv_out_width = specs['width'] - 1  # 2x2 kernel with stride 1
        self.conv_output_features = conv_out_height * conv_out_width * 32

        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_features + n_extra_features, specs['layer_size']),
            nn.ReLU(),
            nn.Linear(specs['layer_size'], specs['layer_size']),
            nn.ReLU(),
            nn.Linear(specs['layer_size'], n_actions)
            # no non-linearity on the last layer, because q values can be anything
        )

    def forward(self, x):
        board_size = self.specs['max_val'] * self.specs['height'] * self.specs['width']

        boards = x[:, :board_size].view(-1, self.specs['height'], self.specs['width'], self.specs['max_val']).float()
        extra_features = x[:,board_size:].float()

        # Permute dimensions to be in (batch_size, max_val, height, width) format
        boards = boards.permute(0, 3, 1, 2)
        
        y = self.conv(boards)
        y = y.reshape(y.shape[0], -1)

        # Concatenate y with extra_features
        z = torch.cat((y, extra_features), dim=1)

        return self.fc(z)

    @classmethod
    def from_file(cls, path, device, specs):
        model = cls(specs)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        return model