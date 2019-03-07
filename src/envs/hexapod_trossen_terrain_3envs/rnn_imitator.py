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
    env = hex_env.Hexapod()
    master = policies.RNN_V2_PG(env)
    optimizer = T.optim.Adam(master.parameters(), lr=3e-4, weight_decay=1e-4)
    lossfun = T.nn.MSELoss()

    # N x EP_LEN x OBS_DIM
    states_A = np.load("A_states.npy")
    acts_A = np.load("A_acts.npy")

    states_B = np.load("A_states.npy")
    acts_B = np.load("A_acts.npy")

    iters = 300
    batchsize = 24

    assert len(states_A) == len(acts_A) == len(states_B) == len(acts_B)
    N_EPS, EP_LEN, OBS_DIM = states_A.shape
    _, _, ACT_DIM = acts_A.shape

    for i in range(iters):
        # Make batch of episodes
        batch_states = []
        batch_acts = []
        for j in range(batchsize):
            rnd_idx = np.random.randint(0, N_EPS)
            A_B_choice = np.random.rand()
            states = states_A[rnd_idx] if A_B_choice < 0.5 else states_B[rnd_idx]
            acts = acts_A[rnd_idx] if A_B_choice < 0.5 else acts_B[rnd_idx]
            batch_states.append(states)
            batch_acts.append(acts)

        batch_states = np.stack(batch_states)
        batch_acts = np.stack(batch_acts)

        assert batch_states.shape == (batchsize, EP_LEN, OBS_DIM)
        assert batch_acts.shape == (batchsize, EP_LEN, ACT_DIM)

        batch_states_T = T.from_numpy(batch_states).float()
        expert_acts_T = T.from_numpy(batch_acts).float()

        # Perform batch forward pass on episodes
        master_acts_T = master.forward_batch(batch_states_T)

        # Update RNN
        N_WARMUP_STEPS = 200
        loss = lossfun(master_acts_T[:, N_WARMUP_STEPS:, :], expert_acts_T[:, N_WARMUP_STEPS:, :])
        loss.backward()
        master.print_info()
        master.soft_clip_grads(0.5)
        optimizer.step()

        # Print info
        if i % 1 == 0:
            print("Iter: {}/{}, loss: {}".format(i, iters, loss))

    if iters == 0:
        master = T.load("master.p")
    else:
        T.save(master, "master.p")

    master.clone_params()

    # Test visually
    while True:
        s = env.reset()
        episode_reward = 0
        h = None
        for i in range(1000):
            act, h = master((my_utils.to_tensor(s, True), h))
            s, r, done, _ = env.step(act[0].detach().numpy())
            episode_reward += r
            env.render()
        print("Episode reward: {}".format(episode_reward))


def imitate_dynamic():

    # Default flat env
    flat_env = hex_env.Hexapod(mem_dim=0, env_name="flat")
    tiles_env = hex_env.Hexapod(mem_dim=0, env_name="tiles")
    rails_env = hex_env.Hexapod(mem_dim=0, env_name="rails")

    master = policies.RNN_S(flat_env)
    optimizer = T.optim.Adam(master.parameters(), lr=3e-4, weight_decay=1e-4)

    expert_flat = T.load('../../algos/PG/agents/Hexapod_RNN_V2_PG_THQ_pg.p')
    expert_tiles = T.load('../../algos/PG/agents/Hexapod_RNN_V2_PG_Z29_pg.p')
    expert_rails = T.load('../../algos/PG/agents/Hexapod_RNN_V2_PG_7NK_pg.p')

    iters = 200
    batchsize = 12

    for i in range(iters):
        # Make batch of episodes
        batch_acts_master = []
        batch_acts_expert = []

        for _ in range(batchsize):
            # Perform rnd_flat flat steps
            s = flat_env.reset()
            h = None
            expert_h = None

            max_steps = 600
            rnd_flat = np.random.randint(120, 200)
            rnd_rest = max_steps - rnd_flat
            for _ in range(rnd_flat):
                action, h = master((my_utils.to_tensor(s, True), h))
                expert_action, expert_h = expert_flat((my_utils.to_tensor(s, True), expert_h))
                s, _, _, _ = flat_env.step(expert_action[0].detach().numpy())
                batch_acts_master.append(action)
                batch_acts_expert.append(expert_action.detach())

            # Randomly choose next env
            rnd_idx = np.random.rand()
            env, expert = (tiles_env, expert_tiles) if rnd_idx < 0.5 else (rails_env, expert_rails)
            expert_h = None

            # Set joint states of hexapod in that env
            last_qpos, last_qvel = flat_env.sim.get_state().qpos.tolist(), flat_env.sim.get_state().qvel.tolist()
            env.set_state(last_qpos, last_qvel)

            # Perform y steps on tiles or rails
            for _ in range(rnd_rest):
                action, h = master((my_utils.to_tensor(s, True), h))
                expert_action, expert_h = expert((my_utils.to_tensor(s, True), expert_h))
                obs, _, _, _, = flat_env.step(expert_action[0].detach().numpy())
                batch_acts_master.append(action)
                batch_acts_expert.append(expert_action.detach())

        batch_acts_master_T = T.cat(batch_acts_master)
        batch_acts_expert_T = T.cat(batch_acts_expert)

        # Update RNN
        N_WARMUP_STEPS = 50
        PEN_MASK = T.ones(max_steps * batchsize)
        for j in range(1, batchsize - 1):
            PEN_MASK[rnd_flat * j :rnd_flat * j + N_WARMUP_STEPS] = 0
        loss = T.mean(T.pow(batch_acts_master_T - batch_acts_expert_T, 2).sum(1) * PEN_MASK)
        loss.backward()
        master.print_info()
        master.soft_clip_grads(0.5)
        optimizer.step()

        # Print info
        if i % 1 == 0:
            print("Iter: {}/{}, loss: {}".format(i, iters, loss))

    # Test visually
    for _ in range(10):
        envs = [flat_env, tiles_env, rails_env]
        for env in envs:
            done = False
            s = env.reset()
            h = None
            episode_reward = 0
            while not done:
                act, h = master(my_utils.to_tensor(s, True), h)
                s, r, done, _ = env.step(act[0].detach().numpy())
                episode_reward += r
                env.render()
            print("Episode reward: {}".format(episode_reward))


def guess_env():

    # Default flat env
    flat_env = hex_env.Hexapod(mem_dim=0, env_name="flat")
    tiles_env = hex_env.Hexapod(mem_dim=0, env_name="tiles")
    rails_env = hex_env.Hexapod(mem_dim=0, env_name="rails")

    master = policies.RNN_CLASSIF(flat_env, 3)
    optimizer = T.optim.Adam(master.parameters(), lr=3e-4, weight_decay=1e-4)
    loss = T.nn.CrossEntropyLoss()

    expert_flat = T.load('../../algos/PG/agents/Hexapod_RNN_V2_PG_THQ_pg.p')
    expert_tiles = T.load('../../algos/PG/agents/Hexapod_RNN_V2_PG_Z29_pg.p')
    expert_rails = T.load('../../algos/PG/agents/Hexapod_RNN_V2_PG_7NK_pg.p')

    iters = 200
    batchsize = 12

    for i in range(iters):
        # Make batch of episodes
        batch_acts_master = []
        batch_acts_expert = []

        for _ in range(batchsize):
            # Perform rnd_flat flat steps
            s = flat_env.reset()
            h = None
            expert_h = None

            max_steps = 600
            rnd_flat = np.random.randint(120, 200)
            rnd_rest = max_steps - rnd_flat
            for _ in range(rnd_flat):
                action, h = master((my_utils.to_tensor(s, True), h))
                expert_action, expert_h = expert_flat((my_utils.to_tensor(s, True), expert_h))
                s, _, _, _ = flat_env.step(expert_action[0].detach().numpy())
                batch_acts_master.append(action)
                batch_acts_expert.append(expert_action.detach())

            # Randomly choose next env
            rnd_idx = np.random.rand()
            env, expert = (tiles_env, expert_tiles) if rnd_idx < 0.5 else (rails_env, expert_rails)
            expert_h = None

            # Set joint states of hexapod in that env
            last_qpos, last_qvel = flat_env.sim.get_state().qpos.tolist(), flat_env.sim.get_state().qvel.tolist()
            env.set_state(last_qpos, last_qvel)

            # Perform y steps on tiles or rails
            for _ in range(rnd_rest):
                action, h = master((my_utils.to_tensor(s, True), h))
                expert_action, expert_h = expert((my_utils.to_tensor(s, True), expert_h))
                obs, _, _, _, = flat_env.step(expert_action[0].detach().numpy())
                batch_acts_master.append(action)
                batch_acts_expert.append(expert_action.detach())

        batch_acts_master_T = T.cat(batch_acts_master)
        batch_acts_expert_T = T.cat(batch_acts_expert)

        # Update RNN
        N_WARMUP_STEPS = 50
        PEN_MASK = T.ones(max_steps * batchsize)
        for j in range(1, batchsize - 1):
            PEN_MASK[rnd_flat * j :rnd_flat * j + N_WARMUP_STEPS] = 0
        loss = T.mean(T.pow(batch_acts_master_T - batch_acts_expert_T, 2).sum(1) * PEN_MASK)
        loss.backward()
        master.print_info()
        master.soft_clip_grads(0.5)
        optimizer.step()

        # Print info
        if i % 1 == 0:
            print("Iter: {}/{}, loss: {}".format(i, iters, loss))

    # Test visually
    for _ in range(10):
        envs = [flat_env, tiles_env, rails_env]
        for env in envs:
            done = False
            s = env.reset()
            h = None
            episode_reward = 0
            while not done:
                act, h = master(my_utils.to_tensor(s, True), h)
                s, r, done, _ = env.step(act[0].detach().numpy())
                episode_reward += r
                env.render()
            print("Episode reward: {}".format(episode_reward))

if __name__=="__main__":
    T.set_num_threads(1)
    guess_env()

