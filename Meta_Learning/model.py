import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_dim, game_type, hidden_dims, conv_filters):
        super(PolicyNetwork, self).__init__()
        self.game_type = game_type
        if game_type == "pong":
            self.conv1 = nn.Conv2d(1, conv_filters[0], kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=4, stride=2)
            self.fc1 = nn.Linear(conv_filters[1] * 7 * 7, 256)
            self.fc2 = nn.Linear(256, output_dim)
        else:
            self.fc1 = nn.Linear(input_shape[0], hidden_dims[0])
            self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if self.game_type == "pong":
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
        return self.softmax(x)