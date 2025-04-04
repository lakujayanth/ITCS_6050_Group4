import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class PatchEmbedCNN(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):  # input_channels=1 because grayscale
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(7 * 7 * 64, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class GeneralizedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        input_channels = observation_space.shape[0]  # e.g., 4 stacked grayscale frames
        self.encoder = PatchEmbedCNN(input_channels=input_channels, output_dim=features_dim)

    def forward(self, observations):
        return self.encoder(observations)

def make_ppo_model(env, action_type='discrete'):
    policy_kwargs = dict(
        features_extractor_class=GeneralizedExtractor,
        net_arch=[128, 128]
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    return model
