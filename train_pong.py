# train_pong.py
import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo
import cv2
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from ppo_framework import make_ppo_model, TrainingTracker
from stable_baselines3 import PPO
import os

class ResizeAndGrayScale(ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = Box(
            low=0, high=255, shape=(self.shape[0], self.shape[1], 1), dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(obs, -1).astype(np.uint8)

def make_pong_env(record=False):
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env = ResizeAndGrayScale(env)
    if record:
        env = RecordVideo(env, video_folder="videos", episode_trigger=lambda ep: True)
    return env

def train_ppo_with_tracking(env, model, total_timesteps, game_name="Pong"):
    tracker = TrainingTracker(game_name)
    timestep = 0
    episode = 0
    
    while timestep < total_timesteps:
        episode += 1
        obs = env.reset()[0]  # Get observation from tuple
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            timestep += 1
            
            obs = next_obs
            
            model.train()  # Trigger training step
            
        # Simplified advantage (actual advantage handled by Stable Baselines3)
        advantage = episode_reward
        # Win if score is positive (opponent didn't score more)
        win_status = episode_reward > 0
        
        tracker.update(episode_reward, episode_steps, advantage, win_status)
        tracker.log_step(timestep, 
                        model.policy.optimizer_loss if hasattr(model.policy, 'optimizer_loss') else 0,
                        model.value_loss if hasattr(model, 'value_loss') else 0,
                        model.entropy_loss if hasattr(model, 'entropy_loss') else 0)
        
        if episode % 10 == 0:
            tracker.print_progress(episode, total_timesteps//episode_steps)
            
    tracker.plot_training_results(save_path="./training_plots")
    return model

if __name__ == "__main__":
    os.makedirs('training_plots', exist_ok=True)
    env = make_pong_env(record=False)
    model = make_ppo_model(env, action_type='discrete')
    model = train_ppo_with_tracking(env, model, total_timesteps=1_000_000)
    model.save("ppo_pong")