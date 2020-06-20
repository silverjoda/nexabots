import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, SAC, TD3
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import SubprocVecEnv
import time
import tensorflow as tf
import sys
import random
import string
import socket


def make_env(env_id, params):
    def _init():
        env = env_id(params["env_list"], max_n_envs=1, specific_env_len=70, s_len=150, walls=True, target_vel=params["target_vel"], use_contacts=params["use_contacts"])
        return env
    return _init


if __name__=="__main__":
    env_list = ["flat"] #  ["flat", "tiles", "triangles", "holes", "pipe", "stairs", "perlin"]

    if len(sys.argv) > 1:
        env_list = [sys.argv[1]]

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"steps": 500000, "batchsize": 60, "gamma": 0.99, "policy_lr": 0.0007, "weight_decay" : 0.0001, "ppo": True,
              "ppo_update_iters": 6, "animate": False, "train" : True, "env_list" : env_list,
              "note" : "...", "ID" : ID, "std_decay" : 0.000, "target_vel" : 0.10, "use_contacts" : True}

    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    from src.envs.hexapod_trossen_terrain_all.hexapod_trossen_limited import Hexapod as env_id

    # TODO: Experiment with RL algo improvement, add VF to PG
    # TODO: Experiment with decayed exploration
    # TODO: Try different RL algos (baselines for example)
    # TODO: Try tiles with contacts and without  (also slow vs fast speed)
    # TODO: Try training with quantized torque on legs
    # TODO: Debug yaw and contacts on real hexapod
    # TODO: Continue wiring up drone
    # TODO: Vary max torque to make policy use feedback

    # Test
    if params["train"]:
        print("Training")
        print(params)
        env = env_id(params["env_list"], max_n_envs=1, specific_env_len=70, s_len=150, walls=True,
                     target_vel=params["target_vel"], use_contacts=params["use_contacts"])

        model = SAC('MlpPolicy', env, learning_rate=3e-3, verbose=1, batch_size=64, tensorboard_log="/tmp", gamma=0.99)
        model.learn(total_timesteps=int(params["steps"]))
        print("Done learning, saving model")
        model.save("agents/SBL_{}".format(params["ID"]))
        print("Saved model, closing env")
        env.close()
        print("Finished training with ID: {}".format(ID))
    else:
        env = env_id(params["env_list"], max_n_envs=1, specific_env_len=70, s_len=150, walls=True,
                     target_vel=params["target_vel"], use_contacts=params["use_contacts"])

        print("Testing")
        policy_name = "H02" # LX3, 63W (tiles): joints + contacts + yaw
        policy_path = 'agents/SBL_{}'.format(policy_name)
        model = SAC.load(policy_path)
        print("Loading policy from: {}".format(policy_path))

        obs = env.reset()
        for _ in range(100):
            cum_rew = 0
            t1 = time.time()
            for i in range(800):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action, render=True)
                cum_rew += reward
                #env.render()
                if done:
                    t2 = time.time()
                    print("Time taken for episode: {}".format(t2-t1))
                    obs = env.reset()
                    print(cum_rew)
                    break

        env.close()
