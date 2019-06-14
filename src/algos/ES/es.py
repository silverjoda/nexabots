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

        #wpen = np.square(w).mean()

        return -reward
    return f


def train(params):
    env, policy, iters, animate, ID = params

    obs_dim, act_dim = env.obs_dim, env.act_dim

    w = parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, 0.7)
    f = f_wrapper(env, policy, animate)

    weight_decay = 0.01

    print("Env: {}, Policy: {}, Action space: {}, observation space: {},"
          " N_params: {}, ID: {}, wd = {}, comments: ...".format(
        env.__class__.__name__, policy.__class__.__name__, act_dim, obs_dim, len(w), ID, weight_decay))

    it = 0
    try:
        while not es.stop():
            it += 1
            if it > iters:
                break
            if it % 30 == 0:
                sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "agents/{}_{}_{}_es.p".format(env.__class__.__name__, policy.__class__.__name__,
                                                                  ID))
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

#from src.envs.cartpole_pbt.cartpole import CartPoleBulletEnv
#env = CartPoleBulletEnv(animate=True)

from src.envs.hexapod_trossen_terrain_all import hexapod_trossen_terrain_all as env
env = env.Hexapod()

policy = policies.NN_PG(env, hid_dim=32)
ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

TRAIN = False

if TRAIN:
    t1 = time.clock()
    train((env, policy, 1000, True, ID))
    t2 = time.clock()
    print("Elapsed time: {}".format(t2 - t1))
else:
    #policy = T.load("agents/CartPoleBulletEnv_NN_PG_GJ1_es.p")
    policy = T.load("agents/Hexapod_NN_PG_458_es.p")
    env.test(policy)

print("Done.")
