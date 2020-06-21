import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C
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
        env = env_id(params["env_list"], max_n_envs=1, specific_env_len=70, s_len=150, walls=True,
                     target_vel=params["target_vel"], use_contacts=params["use_contacts"], turn_dir=params["turn_dir"])
        return env
    return _init


if __name__=="__main__":
    env_list = ["tiles"] #  ["flat", "tiles", "triangles", "holes", "pipe", "stairs", "perlin"]

    # DEPLOY SCHEDULE:
    # Flat ->  turn_dir: None, env_list: flat, note: Flat
    # Rough ->  turn_dir: None, env_list: tiles/perlin, note: Rough
    # Turn left flat -> turn_dir: Left, env_list: flat, note: Turn left flat
    # Turn right flat -> turn_dir: right, env_list: flat, not: Turn right flat
    # Stairs up -> turn_dir: None, env_list: stairs, note: Stairs up
    # Stairs down -> turn_dir: None, env_list: stairs_down, note: Stairs down
    # Stairs turn left -> turn_dir: Left, env_list: stairs, note: Stairs u/d turn left
    # Stairs turn right -> turn_dir: Right, env_list: stairs, note: Stairs u/d turn right

    if len(sys.argv) > 1:
        env_list = [sys.argv[1]]

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"steps": 15000000, "batchsize": 60, "gamma": 0.99, "policy_lr": 0.0007, "weight_decay" : 0.0001, "ppo": True,
              "ppo_update_iters": 6, "animate": True, "train" : False, "env_list" : env_list,
              "note" : "...", "ID" : ID, "std_decay" : 0.000, "target_vel" : 0.10, "use_contacts" : True, "turn_dir" : None}

    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    #from src.envs.hexapod_trossen_terrain_all.hexapod_trossen_limited import Hexapod as env_id
    from src.envs.hexapod_trossen_terrain_all.hexapod_deploy_default import Hexapod as env_id

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
        env = SubprocVecEnv([make_env(env_id, params) for _ in range(4)], start_method='fork')
        policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[96, 96])
        model = A2C('MlpPolicy', env, learning_rate=1e-3, verbose=1, n_steps=64, tensorboard_log="/tmp", gamma=0.99, policy_kwargs=policy_kwargs)
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
        policy_name = "THG" # LX3, 63W (tiles): joints + contacts + yaw
        policy_path = 'agents/SBL_{}'.format(policy_name)
        model = A2C.load(policy_path)
        print("Loading policy from: {}".format(policy_path))

        obs = env.reset()
        env.current_difficulty = 1.
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
