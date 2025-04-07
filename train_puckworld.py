# train_puckworld.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from ppo_framework import make_ppo_model, TrainingTracker
import os

class PuckWorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.puck_pos = np.array([42.0, 42.0])
        self.goal_pos = np.random.uniform(low=10, high=74, size=(2,))
        self.frame = np.zeros((84, 84, 3), dtype=np.uint8)
        self.steps = 0
        self.cumulative_distance = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.puck_pos = np.array([42.0, 42.0])
        self.goal_pos = np.random.uniform(low=10, high=74, size=(2,))
        self.steps = 0
        self.cumulative_distance = 0.0
        self._draw_frame()
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        action = np.clip(action, -1.0, 1.0)
        self.puck_pos += action * 2.0
        self.puck_pos = np.clip(self.puck_pos, 0, 83)
        distance = np.linalg.norm(self.puck_pos - self.goal_pos)
        self.cumulative_distance += distance
        terminated = distance < 5.0
        truncated = self.steps >= 1000
        self._draw_frame()
        obs = self._get_obs()
        if terminated or truncated:
            avg_distance = self.cumulative_distance / self.steps
            reward = -avg_distance
        else:
            reward = 0.0
        return obs, reward, terminated, truncated, {"distance": distance}

    def _draw_frame(self):
        self.frame = np.zeros((84, 84, 3), dtype=np.uint8)
        cv2.circle(self.frame, tuple(self.goal_pos.astype(int)), 3, (0, 0, 255), -1)
        cv2.circle(self.frame, tuple(self.puck_pos.astype(int)), 4, (0, 255, 0), -1)

    def _get_obs(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray, -1).astype(np.uint8)

    def render(self):
        return self.frame

def train_ppo_with_tracking(env, model, total_timesteps, game_name="Puckworld"):
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
        win_status = info.get('distance', float('inf')) < 5.0
        
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
    env = PuckWorldEnv()
    model = make_ppo_model(env, action_type='continuous')
    model = train_ppo_with_tracking(env, model, total_timesteps=500_000)
    model.save("ppo_puckworld")