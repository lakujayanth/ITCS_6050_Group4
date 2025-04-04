from puckworld_env import PuckWorldEnv
from ppo_framework import make_ppo_model

env = PuckWorldEnv()
model = make_ppo_model(env, action_type='continuous')
model.learn(total_timesteps=500_000)
model.save("ppo_puckworld")
