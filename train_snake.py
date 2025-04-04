from snake_env import SnakeEnv
from ppo_framework import make_ppo_model

env = SnakeEnv()
model = make_ppo_model(env, action_type='discrete')
model.learn(total_timesteps=500_000)
model.save("ppo_snake")
