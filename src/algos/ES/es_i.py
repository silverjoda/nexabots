import numpy as np
import cma
from time import sleep
import torch
import torch as T
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import time
import os
import sys

import src.my_utils as my_utils
import src.policies as policies
import random
import string
import socket#


def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        vector_to_parameters(w, policy.parameters())

        while not done:

            # Get action from policy
            with torch.no_grad():
                act = policy(my_utils.to_tensor(obs, True))

            # Step environment
            obs, rew, done, _ = env.step(act.squeeze(0).numpy())

            if animate:
                env.render()

            reward += rew

        return reward
    return f


def train1(env, policy, params):
    w = parameters_to_vector(policy.parameters()).detach()
    f = f_wrapper(env, policy, params["animate"])
    n = len(w)
    sig = T.ones(n)

    for i in range(params["iters"]):

        # Generate noise population
        eps = sig * T.randn(params["popsize"], n)
        population = T.split(w.repeat(params["popsize"], 1) + eps, n, dim=0)

        # Evaluate population
        R = [f(p) for p in population]

        # Update gradients
        grads = (1 / (params["popsize"] * sig)) * T.sum(eps * R)

        # Apply update rule
        w = w + params["learning_rate"] * grads

        if i % 200 == 0:
            sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "agents/{}_{}_{}_es.p".format(env.__class__.__name__, policy.__class__.__name__,
                                                              ID))
            vector_to_parameters(w, policy.parameters())
            T.save(policy, sdir)
            print("Saved checkpoint, {}".format(sdir))

    return w


if __name__=="__main__":
    T.set_num_threads(1)

    env_list = ["flat"]
    if len(sys.argv) > 1:
        env_list = [sys.argv[1]]

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 10000, "popsize": 24, "learning_rate" : 1, "weight_decay" : 0.0003, "animate": True, "train" : True, "env_list" : env_list,
              "note" : "Test", "ID" : ID}

    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    #from src.envs.hexapod_trossen_terrain_all import hexapod_trossen_terrain_all as hex_env
    #env = hex_env.Hexapod(env_list=env_list, max_n_envs=1)

    from src.envs.ctrl_slider.ctrl_slider import SliderEnv
    env = SliderEnv(mass_std=0.3, damping_std=0, animate=params["animate"])

    # Test
    if params["train"]:
        print("Training")
        policy = policies.NN_PG(env, 32, tanh=False, std_fixed=True)
        print(params, env.obs_dim, env.act_dim, env.__class__.__name__, policy.__class__.__name__)

        t1 = time.clock()
        train1(env, policy, params)
        t2 = time.clock()
        print("Elapsed time: {}".format(t2 - t1))
    else:
        print("Testing")

        policy = T.load('agents/SliderEnv_NN_PG_4JX_pg.p')
        env.test(policy)

print("Done.")
