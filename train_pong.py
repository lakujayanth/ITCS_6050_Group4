from pong_env import make_pong_env
from ppo_framework import make_ppo_model

env = make_pong_env()
model = make_ppo_model(env, action_type='discrete')
model.learn(total_timesteps=1_000_000)
model.save("ppo_pong")
