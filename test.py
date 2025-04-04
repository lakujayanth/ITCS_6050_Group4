import gymnasium as gym
import ale_py
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
obs, _ = env.reset()
print(obs.shape)
