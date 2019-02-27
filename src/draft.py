import gym

from stable_baselines.common.policies import MlpPolicy, LstmPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import *

env = gym.make('MemoryEnv-v0')
# Vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = A2C(MlpLnLstmPolicy, env, verbose=1, tensorboard_log="/tmp/a2c_cartpole_tensorboard/")
# Train the agent
model.learn(total_timesteps=1000000)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()