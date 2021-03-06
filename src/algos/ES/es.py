import numpy as np
import cma
from time import sleep
import torch
import torch as T
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import time
import os

import src.my_utils as my_utils
import src.policies as policies
import random
import string

T.set_num_threads(1)

def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        vector_to_parameters(torch.from_numpy(w).float(), policy.parameters())

        while not done:

            # Get action from policy
            with torch.no_grad():
                act = policy(my_utils.to_tensor(obs, True))

            # Step environment
            obs, rew, done, _ = env.step(act.squeeze(0).numpy())

            if animate:
                env.render()

            reward += rew

        return -reward
    return f


def train(params):
    env, policy, iters, animate, ID = params

    w = parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, 0.9)
    f = f_wrapper(env, policy, animate)

    it = 0
    try:
        while not es.stop():
            it += 1
            if it > iters:
                break
            if it % 30 == 0:
                sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "agents/{}_es.p".format(ID))
                vector_to_parameters(torch.from_numpy(es.result.xbest).float(), policy.parameters())
                T.save(policy, sdir)
                print("Saved checkpoint, {}".format(sdir))

            print(es.mean.min(), es.mean.max())
            X = es.ask()

            es.tell(X, [f(x) for x in X])
            es.disp()

    except KeyboardInterrupt:
        print("User interrupted process.")

    return es.result.fbest


from src.envs.hexapod_trossen_terrain_all.hexapod_trossen_limited import Hexapod as env
env = env(["flat"], max_n_envs=1, specific_env_len=70, s_len=100, walls=True, target_vel=0.1, use_contacts=False)

policy = policies.CYC_HEX()
ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

TRAIN = True

if TRAIN:
    t1 = time.clock()
    train((env, policy, 1000, True, ID))
    t2 = time.clock()
    print("Elapsed time: {}".format(t2 - t1))
else:
    policy = T.load("agents/XXX_es.p")
    env.test(policy)

print("Done.")
