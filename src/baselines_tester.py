import gym

from stable_baselines.common.policies import MlpPolicy, LstmPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import *

from src.envs.adaptive_ctrl_env.adaptive_ctrl_env import AdaptiveSliderEnv
from src.envs.hexapod_trossen_adapt.hexapod_trossen_adapt import Hexapod

#env = gym.make('MemoryEnv-v0')
env = Hexapod()
# Vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="/tmp/a2c_cartpole_tensorboard/", n_steps=400)
# Train the agent
model.learn(total_timesteps=300000)
model.save("ppo_adapt")
#model = PPO2.load("ppo_adapt")

# Enjoy trained agent
for i in range(100):
    obs = env.reset()
    for j in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()