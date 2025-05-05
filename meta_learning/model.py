import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_dim, game_type, hidden_dims, conv_filters, frame_stack=1, dropout_rate=0.3):
        super(PolicyNetwork, self).__init__()
        self.game_type = game_type
        if game_type == "pong":
            self.conv1 = nn.Conv2d(frame_stack, conv_filters[0], kernel_size=4, stride=2)
            self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=3, stride=2)

            def conv_output_size(input_size, kernel_size, stride, padding=0):
                return (input_size - kernel_size + 2 * padding) // stride + 1

            h, w = 40, 40
            h = conv_output_size(h, kernel_size=4, stride=2)  # 18
            w = conv_output_size(w, kernel_size=4, stride=2)  # 18
            h = conv_output_size(h, kernel_size=3, stride=2)  # 8
            w = conv_output_size(w, kernel_size=3, stride=2)  # 8
            conv_output_dim = conv_filters[1] * h * w
            logger.info(f"Pong conv output dim: {conv_output_dim} (channels={conv_filters[1]}, h={h}, w={w})")

            self.fc1 = nn.Linear(conv_output_dim, hidden_dims[0])
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        else:
            self.fc1 = nn.Linear(input_shape[0], hidden_dims[0])
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.softmax = nn.Softmax(dim=-1)

        self._debug_init(input_shape)

    def _debug_init(self, input_shape):
        device = next(self.parameters()).device
        if self.game_type == "pong":
            dummy_input = torch.zeros(1, self.conv1.in_channels, 40, 40, dtype=torch.float32).to(device)
        else:
            dummy_input = torch.zeros(1, input_shape[0], dtype=torch.float32).to(device)
        try:
            with torch.no_grad():
                output = self.forward(dummy_input)
            logger.info(f"Successfully initialized {self.game_type} network with output shape {output.shape}")
        except Exception as e:
            logger.error(f"Error during {self.game_type} network initialization: {str(e)}")
            raise e

    def forward(self, x):
        logger.debug(f"Forward pass for {self.game_type}: Input shape {x.shape}, dtype {x.dtype}, device {x.device}")
        if self.game_type == "pong":
            if x.dim() == 3:
                x = x.unsqueeze(0)
            if x.shape[1] != self.conv1.in_channels:
                x = x.permute(0, 3, 1, 2)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
        return self.softmax(x)