# ppo_framework.py
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import os

class PatchEmbedCNN(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):
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
        input_channels = observation_space.shape[0]
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

class TrainingTracker:
    def __init__(self, game_name):
        self.game_name = game_name
        self.history = defaultdict(list)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = []
        self.advantages = []
        
    def update(self, reward, episode_length, advantage, win_status=None):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(episode_length)
        self.advantages.append(advantage)
        if win_status is not None:
            self.wins.append(1 if win_status else 0)
            
    def log_step(self, step, policy_loss, value_loss, entropy):
        self.history['policy_loss'].append(policy_loss)
        self.history['value_loss'].append(value_loss)
        self.history['entropy'].append(entropy)
        self.history['steps'].append(step)
        
    def print_progress(self, episode, total_episodes):
        avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
        avg_advantage = np.mean(self.advantages[-100:]) if self.advantages else 0
        win_rate = np.mean(self.wins[-100:]) * 100 if self.wins else 0
        
        print(f"\nGame: {self.game_name}")
        print(f"Episode {episode}/{total_episodes}")
        print(f"Average Reward (last 100): {avg_reward:.2f}")
        print(f"Average Episode Length (last 100): {avg_length:.2f}")
        print(f"Average Advantage (last 100): {avg_advantage:.2f}")
        if self.wins:
            print(f"Win Rate (last 100): {win_rate:.2f}%")
        print("-" * 50)

    def plot_training_results(self, save_path=None):
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        rewards_smoothed = np.convolve(self.episode_rewards, 
                                     np.ones(100)/100, 
                                     mode='valid')
        plt.plot(rewards_smoothed, label='Smoothed Reward')
        plt.plot(self.episode_rewards, alpha=0.3, label='Raw Reward')
        plt.title(f'{self.game_name} - Reward Curve')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        if self.wins:
            plt.subplot(2, 2, 2)
            win_rate = np.convolve(self.wins, 
                                 np.ones(100)/100, 
                                 mode='valid') * 100
            plt.plot(win_rate)
            plt.title(f'{self.game_name} - Win Rate')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate (%)')
        
        plt.subplot(2, 2, 3)
        advantages_smoothed = np.convolve(self.advantages, 
                                        np.ones(100)/100, 
                                        mode='valid')
        plt.plot(advantages_smoothed, label='Smoothed Advantage')
        plt.plot(self.advantages, alpha=0.3, label='Raw Advantage')
        plt.title(f'{self.game_name} - Advantage')
        plt.xlabel('Episode')
        plt.ylabel('Advantage')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(self.history['policy_loss'], label='Policy Loss')
        plt.plot(self.history['value_loss'], label='Value Loss')
        plt.plot(self.history['entropy'], label='Entropy')
        plt.title(f'{self.game_name} - Training Losses')
        plt.xlabel('Training Step')
        plt.ylabel('Loss Value')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/{self.game_name}_training_plots.png")
        plt.show()