import gym
import time
from stable_baselines.common.policies import MlpPolicy, LstmPolicy, MlpLnLstmPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, ACKTR, SAC, A2C, ACER, DDPG, TRPO

from src.envs.cartpole_pbt.hangpole import HangPoleBulletEnv
from src.envs.cartpole_pbt.cartpole_balance import CartPoleBalanceBulletEnv
#env = HangPoleBulletEnv(animate=False, latent_input=False, action_input=False)
env = CartPoleBalanceBulletEnv(animate=False, latent_input=False, action_input=False)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# def make_env():
#     def _init():
#         env = HangPoleBulletEnv(animate=False, latent_input=False, action_input=False)
#         return env
#     return _init
#
# env = SubprocVecEnv([make_env() for i in range(4)])

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


class LSTMCustom(MlpLnLstmPolicy):
    def __init__(self, *args, **kwargs):
        super(LSTMCustom, self).__init__(*args, **kwargs, n_lstm=8)


class MLPCustom(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MLPCustom, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[12, 12],
                                                          vf=[12, 12])],
                                           feature_extraction="mlp")

model = PPO2(MLPCustom, env, gamma=0.995, learning_rate=5e-4, n_steps=150, verbose=1, nminibatches=1)
model.learn(total_timesteps=800000)
model.save("model")

[e.kill() for e in env.unwrapped.envs]
del env

env = CartPoleBalanceBulletEnv(animate=True, latent_input=False, action_input=False)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

for _ in range(100):
    obs = env.reset()
    for i in range(400):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        time.sleep(0.01)
        env.render()

env.close()

