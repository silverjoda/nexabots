import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env
import time
import sys
import random
import string
import socket

if __name__=="__main__":

    env_list = ["flat"] #  ["flat", "tiles", "triangles", "holes", "pipe", "stairs", "perlin"]

    if len(sys.argv) > 1:
        env_list = [sys.argv[1]]

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"steps": 200000, "batchsize": 60, "gamma": 0.995, "policy_lr": 0.0007, "weight_decay" : 0.0001, "ppo": True,
              "ppo_update_iters": 6, "animate": False, "train" : True, "env_list" : env_list,
              "note" : "Straight line with yaw", "ID" : ID, "std_decay" : 0.000, "target_vel" : 0.10, "use_contacts" : True}

    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    from src.envs.hexapod_trossen_terrain_all.hexapod_trossen_limited import Hexapod as env
    env = env(env_list, max_n_envs=1, specific_env_len=70, s_len=100, walls=True, target_vel=params["target_vel"], use_contacts=params["use_contacts"])

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
        model = A2C('MlpPolicy', env, learning_rate=1e-3, verbose=1)
        model.learn(total_timesteps=int(params["steps"]))
        model.save("agents/SBL_{}".format(params["ID"]))
        env.close()
    else:
        print("Testing")
        policy_name = "7JY" # LX3: joints + contacts + yaw
        policy_path = 'agents/SBL_{}'.format(policy_name)
        model = A2C.load(policy_path)
        print("Loading policy from: {}".format(policy_path))

        #print(evaluate_policy(model, env, n_eval_episodes=3))

        obs = env.reset()
        for _ in range(100):
            cum_rew = 0
            for i in range(800):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action, render=True)
                cum_rew += reward
                #env.render()
                if done:
                    obs = env.reset()
                    print(cum_rew)
                    break

        env.close()