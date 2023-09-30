import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, max_val, board_height, board_width, layer_size, n_actions):
        super(DQN, self).__init__()

        # self.input_features = max_val * board_height * board_width
        self.input_features = 5 * max_val * board_height * board_width + 4 * 3
        # actual board result and 4 * possible next state
        # for each possible next state: done, reward, valid 

        self.fc = nn.Sequential(
            nn.Linear(self.input_features, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, n_actions)
            # no non-linearity on the last layer, because q values can be anything
        )

    def forward(self, x):
        input = x.view(x.shape[0], -1).float()  
        return self.fc(input)
    
    @classmethod
    def from_file(cls, path, device, max_val, board_height, board_width, layer_size, n_actions):
        model = cls(max_val, board_height, board_width, layer_size, n_actions)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        return model
    
# alterntaive network. TODO test    
class DQNConv(nn.Module):
    def __init__(self, max_val, board_height, board_width, layer_size, n_actions):
        super(DQNConv, self).__init__()

        # 2D Convolutional layer
        self.conv = nn.Conv2d(in_channels=max_val, out_channels=layer_size, kernel_size=2, stride=1)
        
        # Calculate the output dimensions of the convolutional layer
        conv_out_height = board_height - 1  # 2x2 kernel with stride 1
        conv_out_width = board_width - 1  # 2x2 kernel with stride 1
        self.conv_output_features = conv_out_height * conv_out_width * layer_size

        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_features, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, n_actions)
            # no non-linearity on the last layer, because q values can be anything
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float()  # Rearrange from NHWC to NCHW
        x = self.conv(x)
        x = x.view(x.shape[0], -1)  # Flatten
        return self.fc(x)