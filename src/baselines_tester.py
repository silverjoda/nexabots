import gym
from gym import spaces

from stable_baselines.common.policies import MlpPolicy, LstmPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import *

from src.envs.cartpole_pbt.cartpole_variable import CartPoleBulletEnv

# Vectorized environments allow to easily multiprocess training
# we demonstrate its usefulness in the next examples
n_cpu = 4
env = SubprocVecEnv([lambda: CartPoleBulletEnv(animate=False, latent_input=False, action_input=False) for i in range(n_cpu)])

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="/tmp/ppo2_terrain/", n_steps=400)
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