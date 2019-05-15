import gym

from stable_baselines.common.policies import MlpPolicy, LstmPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import *

from src.envs.ctrl_slider.adaptive_ctrl_env import AdaptiveSliderEnv
from src.envs.hexapod_trossen_adapt.hexapod_trossen_adapt import Hexapod
from src.envs.hexapod_flat_pd_mjc.hexapod_pd import Hexapod

#env = gym.make('MemoryEnv-v0')
env = Hexapod()

# Vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="/tmp/ppo2_terrain/", n_steps=256)
# Train the agent
model.learn(total_timesteps=1500000)
model.save("ppo_terrain")

#model = A2C.load("a2c_adapt")

# Enjoy trained agent
for i in range(100):
    obs = env.reset()
    for j in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()