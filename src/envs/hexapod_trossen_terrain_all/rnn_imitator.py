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
from src.envs.hexapod_trossen_terrain_all import hexapod_trossen_terrain_all as hex_env

def imitate_static():
    env = hex_env.Hexapod()
    master = policies.RNN_V3_PG(env, hid_dim=128, memory_dim=128, n_temp=3).cuda()
    optimizer = T.optim.Adam(master.parameters(), lr=3e-4)
    lossfun = T.nn.MSELoss()

    # N x EP_LEN x OBS_DIM
    expert_states = np.load("data/states_B.npy")
    # N x EP_LEN x ACT_DIM
    expert_acts = np.load("data/acts_B.npy")

    iters = 20000
    batchsize = 32

    assert len(expert_states) == len(expert_acts)
    N_EPS, EP_LEN, OBS_DIM = expert_states.shape
    _, _, ACT_DIM = expert_acts.shape

    for i in range(iters):
        # Make batch of episodes
        batch_states = []
        batch_acts = []
        for _ in range(batchsize):
            rnd_idx = np.random.randint(0, N_EPS)
            states = expert_states[rnd_idx]
            acts = expert_acts[rnd_idx]
            batch_states.append(states)
            batch_acts.append(acts)

        batch_states = np.stack(batch_states)
        batch_acts = np.stack(batch_acts)

        assert batch_states.shape == (batchsize, EP_LEN, OBS_DIM)
        assert batch_acts.shape == (batchsize, EP_LEN, ACT_DIM)

        batch_states_T = T.from_numpy(batch_states).float().cuda()
        expert_acts_T = T.from_numpy(batch_acts).float().cuda()

        # Perform batch forward pass on episodes
        master_acts_T, _ = master.forward((batch_states_T, None))

        # Update RNN
        N_WARMUP_STEPS = 20
        loss = lossfun(master_acts_T[:, N_WARMUP_STEPS:, :], expert_acts_T[:, N_WARMUP_STEPS:, :])
        loss.backward()
        master.soft_clip_grads()
        optimizer.step()

        # Print info
        if i % 10 == 0:
            print("Iter: {}/{}, loss: {}".format(i, iters, loss))

    master = master.cpu()
    T.save(master, "master_B.p")
    print("Done")


def test():
    env = hex_env.Hexapod(mem_dim=0)
    master = T.load("master_B.p", map_location='cpu')

    # Test visually
    while True:
        done = False
        s = env.reset(init_pos=(np.random.rand() + 0.25, 0, 0.1))
        h = None
        episode_reward = 0
        with T.no_grad():
            for i in range(1000):
                act, h = master((my_utils.to_tensor(s, True).unsqueeze(0), h))
                s, r, done, _ = env.step(act[0][0].numpy())
                episode_reward += r
                env.render()
        print("Episode reward: {}".format(episode_reward))


def imitate_dynamic():
    env = hex_env.Hexapod(mem_dim=0)
    master = policies.RNN_PG(env)
    optimizer = T.optim.Adam(master.parameters(), lr=1e-3)
    loss = T.nn.MSELoss()

    # N x EP_LEN x OBS_DIM
    states_A = np.load("states_A.npy")
    # N x EP_LEN x ACT_DIM
    acts_A = np.load("acts_A.npy")

    states_B = np.load("states_B.npy")
    acts_B = np.load("acts_B.npy")

    iters = 1000
    batchsize = 32

    assert len(states_A) == len(acts_A) == len(states_B) == len(acts_B)
    N_EPS, EP_LEN, OBS_DIM = states_A.shape
    _, _, ACT_DIM = acts_A.shape

    for i in range(iters):
        # Make batch of episodes
        batch_states = []
        batch_acts = []
        for i in range(batchsize):
            rnd_idx = np.random.randint(0, N_EPS)
            A_B_choice = np.random.rand()
            states = states_A[rnd_idx] if A_B_choice < 0.5 else states_B[rnd_idx]
            acts = acts_A[rnd_idx] if A_B_choice < 0.5 else acts_B[rnd_idx]
            batch_states.append(states)
            batch_acts.append(acts)

        batch_states = np.concatenate(batch_states)
        batch_acts = np.concatenate(batch_acts)

        assert batch_states.shape == (batchsize, EP_LEN, OBS_DIM)
        assert batch_acts.shape == (batchsize, EP_LEN, ACT_DIM)

        batch_states_T = T.from_numpy(batch_states)
        expert_acts_T = T.from_numpy(batch_acts)

        # Perform batch forward pass on episodes
        master_acts_T = master.forward_batch(batch_states_T)

        # Update RNN
        N_WARMUP_STEPS = 20
        loss = loss(master_acts_T[:, N_WARMUP_STEPS:, :], expert_acts_T[:, N_WARMUP_STEPS:, :])
        loss.backward()
        optimizer.step()

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


if __name__=="__main__":
    T.set_num_threads(1)
    imitate_static()

