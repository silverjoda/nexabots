import os
import sys

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import time
import src.my_utils as my_utils
import src.policies as policies
import random
import string
import socket
from src.envs.hexapod_trossen_terrain import hexapod_trossen_terrain as hex_env

if __name__=="__main__":
    T.set_num_threads(1)

    env = hex_env.Hexapod(mem_dim=0)
    master = policies.RNN_PG(env)
    optimizer = T.optim.Adam(master.parameters(), lr=1e-3)
    loss = T.nn.MSELoss()

    states_A = np.load("states_A.npy")
    acts_A = np.load("acts_A.npy")

    states_B = np.load("states_B.npy")
    acts_B = np.load("acts_B.npy")

    iters = 1000
    batchsize = 64

    assert len(states_A) == len(acts_A) == len(states_B) == len(acts_B)
    N_episodes = len(states_A)

    for i in range(iters):
        # Make batch of episodes
        batch_states = []
        batch_acts = []
        for i in range(batchsize):
            rnd_idx = np.random.randint(0, N_episodes)
            states = states_A[rnd_idx] if np.random.rand() < 0.5 else states_B[rnd_idx]
            acts = acts_A[rnd_idx] if np.random.rand() < 0.5 else acts_B[rnd_idx]
            batch_states.append(states)
            batch_acts.append(acts)

        batch_states = np.concatenate(batch_states)
        batch_acts = np.concatenate(batch_acts)

        batch_states_T = T.from_numpy(batch_states)
        expert_acts_T = T.from_numpy(batch_acts)

        # Perform batch forward pass on episodes
        master_acts = master.forward_batch(batch_states)

        # Update RNN
        loss = loss(master_acts, expert_acts_T)

        # Print info
        if i % 10 == 0:
            print("Iter: {}/{}, loss: {}".format(i, iters, loss))

    # Test visually
    while True:
        done = False
        s = env.reset()
        episode_reward = 0
        while not done:
            act = master(my_utils.to_tensor(s, True))[0].detach().numpy()
            s, r, done, _ = env.step(act)
            episode_reward += r
        print("Episode reward: {}".format(episode_reward))

