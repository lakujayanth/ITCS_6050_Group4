# train_snake.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from ppo_framework import make_ppo_model, TrainingTracker
import os

class SnakeEnv(gym.Env):
    metadata = {"render_mode": "rgb_array"}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.frame = np.zeros((84, 84, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.frame = np.zeros((84, 84, 3), dtype=np.uint8)
        obs = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(obs, -1), {}

    def step(self, action):
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        obs = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(obs, -1), reward, terminated, truncated, info

    def render(self):
        return self.frame

def train_ppo_with_tracking(env, model, total_timesteps, game_name="Snake"):
    tracker = TrainingTracker(game_name)
    timestep = 0
    episode = 0
    
    while timestep < total_timesteps:
        episode += 1
        obs = env.reset()[0]
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
            
            model.train()
            
        advantage = episode_reward
        win_status = episode_reward > 0  # Customize based on actual Snake win condition
        
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
    env = SnakeEnv()
    model = make_ppo_model(env, action_type='discrete')
    model = train_ppo_with_tracking(env, model, total_timesteps=500_000)
    model.save("ppo_snake")