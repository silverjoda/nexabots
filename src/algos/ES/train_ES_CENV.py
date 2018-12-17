import numpy as np
import cma
from time import sleep
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import time
import mujoco_py

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing as mp
import os
from copy import deepcopy

class LinearPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(LinearPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        x = T.tanh(self.fc1(x))
        return x


class NN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 32)
        self.fc2 = nn.Linear(32, 12)
        self.fc3 = nn.Linear(12, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = T.tanh(self.fc3(x))
        return x


def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs, _ = env.reset()

        vector_to_parameters(torch.from_numpy(w).float(), policy.parameters())

        while not done:

            # Get action from policy
            with torch.no_grad():
                act = policy(torch.from_numpy(np.expand_dims(obs, 0)))[0].numpy()

            # Step environment
            obs, rew, done, _ = env.step(act)

            if animate:
                env.render()

            reward += rew

        return -reward
    return f


def f_mp(pos, env_fun, sim, policy, w, output):
    env = env_fun(animate=False, sim=sim)

    reward = 0
    done = False
    obs, _ = env.reset()

    vector_to_parameters(torch.from_numpy(w).float(), policy.parameters())

    while not done:

        # Get action from policy
        with torch.no_grad():
            act = policy(torch.from_numpy(np.expand_dims(obs, 0)))[0].numpy()

        # Step environment
        obs, rew, done, _ = env.step(act)

        reward += rew

    output.put((pos, -reward))


def train(params):
    env_fun, iters, n_hidden, animate, _ = params

    env = env_fun(animate)
    obs_dim, act_dim = env.obs_dim, env.act_dim
    policy = NN(obs_dim, act_dim).float()
    w = parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, 0.5)
    f = f_wrapper(env, policy, animate)

    print("Env: {} Action space: {}, observation space: {}, N_params: {}, comments: ...".format(env_fun.__name__, act_dim,
                                                                                                obs_dim, len(w)))

    it = 0
    try:
        while not es.stop():
            it += 1
            if it > iters:
                break
            if it % 1000 == 0:
                T.save(policy, os.path.join(os.getcwd(), "agents/{}.p".format(env_fun.__name__)))
                print("Saved checkpoint")
            X = es.ask()
            es.tell(X, [f(x) for x in X])
            es.disp()
    except KeyboardInterrupt:
        print("User interrupted process.")

    return es.result.fbest


def train_mt(params):
    env_fun, iters, n_hidden, animate, model = params
    env = env_fun()
    obs_dim, act_dim = env.obs_dim, env.act_dim

    policy = NN(obs_dim, act_dim).float()
    w = parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, 0.5)

    print("Env: {} Action space: {}, observation space: {}, N_params: {}, comments: ...".format("Ant_reach", act_dim,
                                                                                              obs_dim, len(w)))

    sims = [mujoco_py.MjSim(model) for _ in range(es.popsize)]
    policies = [policy] * es.popsize

    ctr = 0
    try:
        while not es.stop():
            ctr += 1
            if ctr > iters:
                break
            if ctr % 1000 == 0:
                sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "agents/{}.p".format(env_fun.__name__))
                T.save(policy, sdir)
                print("Saved checkpoint")
            X = es.ask()

            output = mp.Queue()
            processes = [mp.Process(target=f_mp, args=(i, env_fun, sim, policy, x, output))
                         for i, env_fun, sim, policy, x in zip(range(es.popsize), [env_fun] * es.popsize, sims, policies, X)]

            # Run processes
            for p in processes:
                p.start()

            # Exit the completed processes
            for p in processes:
                p.join()

            evals = [output.get() for _ in processes]

            es.tell(X, evals)
            es.disp()
    except KeyboardInterrupt:
        print("User interrupted process.")

    return es.result.fbest

from src.envs.ant_terrain_mjc.ant_terrain_mjc import AntTerrainMjc
modelpath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "../../envs/ant_terrain_mjc/assets/ant_terrain_mjc.xml")
model = mujoco_py.load_model_from_path(modelpath)
T.set_num_threads(1)

env = AntTerrainMjc # ll
t1 = time.clock()
train_mt((env, 100, 7, False, model))
t2 = time.clock()
print("Elapsed time: {}".format(t2 - t1))



