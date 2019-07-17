import gym
import time
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, ACKTR, SAC, A2C

from src.envs.cartpole_pbt.hangpole import HangPoleBulletEnv
env = HangPoleBulletEnv(animate=False, latent_input=False, action_input=False)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# def make_env():
#     def _init():
#         env = CartPoleBulletEnv(animate=False, latent_input=False, action_input=False)
#         return env
#     return _init
#
# env = SubprocVecEnv([make_env() for i in range(4)])

model = PPO2(MlpPolicy, env, verbose=1, n_steps=300)
model.learn(total_timesteps=1000000)

[e.kill() for e in env.unwrapped.envs]
del env

env = HangPoleBulletEnv(animate=True, latent_input=False, action_input=False)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

for _ in range(100):
    obs = env.reset()
    for i in range(400):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        time.sleep(0.01)
        env.render()

env.close()

# TODO: Try PPO and PPO with LSTM on hangpole_po