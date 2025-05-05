# pong_env.py
import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo
import cv2
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

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
