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
from src.envs.hexapod_trossen_terrain_3envs import hexapod_trossen_terrain_3envs as hex_env

def imitate_static():
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


def imitate_dynamic():

    # Default flat env
    flat_env = hex_env.Hexapod(mem_dim=0, env_name="flat")
    tiles_env = hex_env.Hexapod(mem_dim=0, env_name="tiles")
    rails_env = hex_env.Hexapod(mem_dim=0, env_name="rails")

    master = policies.RNN_PG(flat_env)
    optimizer = T.optim.Adam(master.parameters(), lr=1e-3)
    loss = T.nn.MSELoss()

    expert_flat = T.load('agents/Hexapod_RNN_V2_PG_xxx_pg.p')
    expert_tiles = T.load('agents/Hexapod_RNN_V2_PG_xxx_pg.p')
    expert_rails = T.load('agents/Hexapod_RNN_V2_PG_xxx_pg.p')

    iters = 1000
    batchsize = 32

    for i in range(iters):
        # Make batch of episodes
        batch_states = []
        batch_acts = []
        for i in range(batchsize):

            # Perform rnd_flat flat steps
            s = flat_env.reset()
            h = None

            rnd_flat = np.random.randint(120, 200)
            rnd_rest = 600 - rnd_flat
            for i in range(rnd_flat):
                action, h = master((my_utils.to_tensor(s, True), h))
                expert_action, expert_h = expert_flat((my_utils.to_tensor(s, True), h))
                obs, _, _, _ = flat_env.step(action[0].detach().numpy())

            # Randomly choose next env
            rnd_idx = np.random.rand()
            env, expert = (tiles_env, expert_tiles) if rnd_idx < 0.5 else (rails_env, expert_rails)

            # Set joint states of hexapod in that env


            # Perform y steps on tiles or rails
            for i in range(rnd_rest):
                action, h = master((my_utils.to_tensor(s, True), h))
                obs, _, _, _, = flat_env.step(action[0].detach().numpy())
                expert_action, expert_h = expert((my_utils.to_tensor(s, True), h))



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

