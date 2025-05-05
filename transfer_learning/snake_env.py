import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class SnakeEnv(gym.Env):
    metadata = {"render_mode": "rgb_array"}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.frame = np.zeros((84, 84, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.frame = np.zeros((84, 84, 3), dtype=np.uint8)
        obs = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(obs, -1), {}

    def step(self, action):
        reward = 1.0  # placeholder
        terminated = False
        truncated = False
        info = {}
        obs = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(obs, -1), reward, terminated, truncated, info

    def render(self):
        return self.frame
