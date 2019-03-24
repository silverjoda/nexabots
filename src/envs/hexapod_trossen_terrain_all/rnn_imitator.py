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


def make_dataset(self, env_list, expert_dict, ID, N, n_envs):
    env = hex_env.Hexapod(env_list)
    max_steps = n_envs * 200

    episode_states = []
    episode_labels = []
    episode_acts = []
    ctr = 0
    while ctr < N:

        # Print info
        print("Iter: {}".format(ctr))

        # Generate new environment
        envs, size_list, scaled_indeces_list = env.generate_hybrid_env(n_envs, max_steps)

        cr = 0
        states = []
        acts = []
        labels = []

        current_env_idx = 0
        current_env = envs[current_env_idx]
        policy = expert_dict[current_env]

        s = self.reset()
        for j in range(max_steps):
            x = self.sim.get_state().qpos.tolist()[0]

            if x > scaled_indeces_list[current_env_idx]:
                current_env_idx += 1
                current_env = env_list[current_env_idx]
                policy = expert_dict[current_env]
                print("Policy switched to {} policy".format(env_list[current_env_idx]))

            states.append(s)
            labels.append(env_list.index(current_env))
            action = policy(my_utils.to_tensor(s, True)).detach()[0].numpy()
            acts.append(action)
            s, r, done, od, = self.step(action)
            cr += r

            self.render()

        # if cr < 50:
        #     continue
        # ctr += 1

        episode_states.append(np.stack(states))
        episode_labels.append(np.stack(labels))
        episode_acts.append(np.stack(acts))

        print("Total episode reward: {}".format(cr))

    np_states = np.stack(episode_states)
    np_acts = np.stack(episode_acts)

    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/states_{}.npy".format(ID)), np_states)
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/acts_{}.npy".format(ID)), np_acts)


def imitate_experts():#
    env = hex_env.Hexapod()
    master = policies.RNN_V3_PG(env, hid_dim=128, memory_dim=128, n_temp=3).cuda()
    optimizer = T.optim.Adam(master.parameters(), lr=3e-4)
    lossfun = T.nn.MSELoss()

    # N x EP_LEN x OBS_DIM
    expert_states = np.load("data/states_C.npy")
    # N x EP_LEN x ACT_DIM
    expert_acts = np.load("data/acts_C.npy")

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
    T.save(master, "master_C.p")
    print("Done")


def imitate_static():#
    env = hex_env.Hexapod()
    master = policies.RNN_V3_PG(env, hid_dim=128, memory_dim=128, n_temp=3).cuda()
    optimizer = T.optim.Adam(master.parameters(), lr=3e-4)
    lossfun = T.nn.MSELoss()

    # N x EP_LEN x OBS_DIM
    expert_states = np.load("data/states_C.npy")
    # N x EP_LEN x ACT_DIM
    expert_acts = np.load("data/acts_C.npy")

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
    T.save(master, "master_C.p")
    print("Done")


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


def test():
    env = hex_env.Hexapod(mem_dim=0)
    #env.env_list = "flat"
    master = T.load("master_B.p", map_location='cpu')

    # Test visually
    while True:
        done = False
        s = env.reset(init_pos=(np.random.rand() * 1 + 0.15, 0, 0.1))
        h = None
        episode_reward = 0
        with T.no_grad():
            for i in range(1000):
                act, h = master((my_utils.to_tensor(s, True).unsqueeze(0), h))
                s, r, done, _ = env.step(act[0][0].numpy())
                episode_reward += r
                env.render()
        print("Episode reward: {}".format(episode_reward))


if __name__=="__main__":
    T.set_num_threads(1)
    imitate_experts()

