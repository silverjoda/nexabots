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

#T.set_num_threads(1)

def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        vector_to_parameters(torch.from_numpy(w).float(), policy.parameters())

        while not done:

            # Get action from policy
            with torch.no_grad():
                act = policy(T.tensor(obs).unsqueeze(0))

            # Step environment
            obs, rew, done, _ = env.step(act.numpy())

            if animate:
                env.render()

            reward += rew

        return -reward
    return f


def train(params):
    env, policy, iters, animate, ID = params

    w = parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, 0.5)
    f = f_wrapper(env, policy, animate)

    it = 0
    try:
        while not es.stop():
            it += 1
            if it > iters:
                break
            if it % 10 == 0:
                sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "agents/{}_es.p".format(ID))
                vector_to_parameters(torch.from_numpy(es.result.xbest).float(), policy.parameters())
                T.save(policy, sdir)
                print("Saved checkpoint, {}".format(sdir))

                #print(es.result.xbest)
                print("Weight: ", es.mean.min(), es.mean.max())
            X = es.ask()

            es.tell(X, [f(x) for x in X])
            es.disp()

    except KeyboardInterrupt:
        print("User interrupted process.")

    return es.result.xbest


from src.envs.hexapod_trossen_terrain_all.hexapod_trossen_cyc import Hexapod as env
env = env(["flat"], max_n_envs=1, specific_env_len=70, s_len=400, walls=True, target_vel=0.20, use_contacts=False)

policy = policies.CYC_HEX_BS()
#policy = policies.CYC_HEX_NN(4)
ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

print("Policy: %s" % policy.__class__.__name__)

TRAIN = "F"

if TRAIN == "T":
    t1 = time.time()
    sol = train((env, policy, 1000, False, ID))
    t2 = time.time()
    print("Elapsed time: {}".format(t2 - t1))
else:
    policy = T.load("agents/L4G_es.p")
    print(list(policy.parameters()))
    env.test(policy, render=True)

print("Done.")
