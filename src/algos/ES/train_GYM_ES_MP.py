import numpy as np
import cma
from time import sleep
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import time
from src.envs.ant_reach.ant_reach import AntReach
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import os


class LinearPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(LinearPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        x = self.fc1(x)
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
        x = self.fc3(x)
        return x


def f_mp(args):
    env_name, policy, w = args
    env = gym.make(env_name)
    reward = 0
    done = False
    obs = env.reset()

    vector_to_parameters(torch.from_numpy(w).float(), policy.parameters())

    while not done:

        # Get action from policy
        with torch.no_grad():
            act = policy(torch.from_numpy(np.expand_dims(obs.astype(np.float32()), 0)))[0].numpy()

        # Step environment
        obs, rew, done, _ = env.step(act)

        reward += rew

    return -reward


def train_mt(params):
    env_name, iters, n_hidden = params

    env = gym.make(env_name)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    policy = NN(obs_dim, act_dim).float()
    w = parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, 0.5)

    print("Env: {} Action space: {}, observation space: {}, N_params: {}, comments: ...".format("Ant_reach", act_dim,
                                                                                                obs_dim, len(w)))
    ctr = 0
    try:
        while not es.stop():
            ctr += 1
            if ctr > iters:
                break
            X = es.ask()

            N = len(X)
            p = Pool(8)

            evals = p.map(f_mp, list(zip([env_name] * N, [policy] * N,  X)))

            es.tell(X, evals)
            es.disp()
    except KeyboardInterrupt:
        print("User interrupted process.")

    return es.result.fbest

import gym
import gym.spaces
T.set_num_threads(1)
gym.logger.MIN_LEVEL = 40

env_name = "AntMinimal-v0"
t1 = time.clock()
train_mt((env_name, 10, 7))
t2 = time.clock()
print("Elapsed time: {}".format(t2 - t1))



