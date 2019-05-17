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
import socket #
from src.envs.hexapod_trossen_terrain_all import hexapod_trossen_terrain_all as hex_env


def make_dataset_rnn_experts(env_list, expert_dict, N, n_envs, render=False):
    env = hex_env.Hexapod(env_list, max_n_envs=n_envs)
    length = n_envs * env.s_len
    env.env_change_prob = 0.1
    change_prob = 0.01

    h = None

    episode_states = []
    episode_labels = []
    ctr = 0
    while ctr < N:

        # Print info
        print("Iter: {}".format(ctr))

        # Generate new environment
        envs, size_list, scaled_indeces_list = env.generate_hybrid_env(n_envs, length)
        scaled_indeces_list.append(length)

        cr = 0
        states = []
        labels = []

        current_env_idx = 0
        current_env = envs[current_env_idx]
        policy = expert_dict[current_env]

        print(envs, scaled_indeces_list, current_env)

        s = env.reset()
        for j in range(int(env.max_steps * 1.5)):
            x = env.sim.get_state().qpos.tolist()[0] * 100 + 20

            if x > scaled_indeces_list[current_env_idx]:
                current_env_idx += 1
                current_env = envs[current_env_idx]
                h = None
                print("Env switched to {} ".format(envs[current_env_idx]))

            if np.random.rand() < change_prob:
                policy_choice = np.random.choice(env_list, 1)[0]
                policy = expert_dict[policy_choice]
                print("Policy switched to {} policy".format(policy_choice))

            states.append(s)
            labels.append(env_list.index(current_env))

            action, h = policy((my_utils.to_tensor(s, True).unsqueeze(0), h))
            action = action[0][0].detach().numpy()
            s, r, done, od, = env.step(action)
            cr += r

            if render:
                env.render()

        if cr < 100:
            continue
        ctr += 1

        episode_states.append(np.stack(states))
        episode_labels.append(np.stack(labels))

        print("Total episode reward: {}".format(cr))

    np_states = np.stack(episode_states)
    np_labels = np.stack(episode_labels)

    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/states.npy"), np_states)
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/labels.npy"), np_labels)


def make_dataset_reactive_experts(env_list, expert_dict, N, n_envs, render=False):
    env = hex_env.Hexapod(env_list, max_n_envs=n_envs)
    length = n_envs * 200
    change_prob = 0.01
    env.env_change_prob = 0.1

    episode_states = []
    episode_labels = []
    ctr = 0
    while ctr < N:

        # Print info
        print("Iter: {}".format(ctr))

        # Generate new environment
        envs, size_list, scaled_indeces_list = env.generate_hybrid_env(n_envs, length)
        scaled_indeces_list.append(length)

        cr = 0
        states = []
        labels = []

        current_env_idx = 0
        current_env = envs[current_env_idx]
        policy = expert_dict[current_env]

        print(envs, scaled_indeces_list, current_env)

        s = env.reset()
        for j in range(n_envs * 200):
            x = env.sim.get_state().qpos.tolist()[0] * 100 + 20

            if x > scaled_indeces_list[current_env_idx]:
                current_env_idx += 1
                current_env = envs[current_env_idx]
                print("Env switched to {} ".format(envs[current_env_idx]))

            if np.random.rand() < change_prob:
                policy_choice = np.random.choice(env_list, 1)[0]
                policy = expert_dict[policy_choice]
                print("Policy switched to {} policy".format(policy_choice))

            states.append(s)
            labels.append(env_list.index(current_env))
            action = policy((my_utils.to_tensor(s, True)))
            action = action[0].detach().numpy()
            s, r, done, od, = env.step(action)
            cr += r

            if render:
                env.render()

        if cr < 100:
            continue
        ctr += 1

        episode_states.append(np.stack(states))
        episode_labels.append(np.stack(labels))

        print("Total episode reward: {}".format(cr))

    np_states = np.stack(episode_states)
    np_labels = np.stack(episode_labels)

    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/states.npy"), np_states)
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/labels.npy"), np_labels)


def train_classifier(n_classes, iters, env_list):
    env = hex_env.Hexapod(env_list, max_n_envs=len(env_list))
    classifier = policies.RNN_CLASSIF_ENV(env, hid_dim=24, memory_dim=24, n_temp=2, n_classes=n_classes, to_gpu=True).cuda()
    optimizer_classifier = T.optim.Adam(classifier.parameters(), lr=2e-4, weight_decay=0.001)
    lossfun_classifier = T.nn.CrossEntropyLoss()

    # N x EP_LEN x OBS_DIM
    expert_states = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/states.npy"))

    # N x EP_LEN x N_CLASSES
    expert_labels = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/labels.npy"))

    batchsize = 32

    N_EPS, EP_LEN, OBS_DIM = expert_states.shape

    for i in range(iters):
        # Make batch of episodes
        batch_states = []
        batch_labels = []
        for _ in range(batchsize):
            rnd_idx = np.random.randint(0, N_EPS - 32)
            states = expert_states[rnd_idx]
            labels = expert_labels[rnd_idx]
            batch_states.append(states)
            batch_labels.append(labels)

        batch_states = np.stack(batch_states)
        batch_labels = np.stack(batch_labels)

        assert batch_states.shape == (batchsize, EP_LEN, OBS_DIM)

        batch_states_T = T.from_numpy(batch_states).float().cuda()
        expert_labels_T = T.from_numpy(batch_labels).long().cuda()

        # Perform batch forward pass on episodes
        labels_T, _ = classifier.forward((batch_states_T, None))

        # Update RNN
        N_WARMUP_STEPS = 0

        trn_loss_clasifier = lossfun_classifier(labels_T[:, N_WARMUP_STEPS:].contiguous().view(-1, n_classes), expert_labels_T[:, N_WARMUP_STEPS:].contiguous().view(-1))
        trn_loss_clasifier.backward()
        classifier.soft_clip_grads()
        optimizer_classifier.step()

        # Print info
        if i % 10 == 0:
            states = expert_states[-32:]
            labels = expert_labels[-32:]
            batch_states_T = T.from_numpy(states).float().cuda()
            expert_labels_T = T.from_numpy(labels).long().cuda()
            labels_T, _ = classifier.forward((batch_states_T, None))
            pred = T.argmax(labels_T.contiguous().view(-1, n_classes), dim=1)
            labs = expert_labels_T.contiguous().view(-1)
            tst_accuracy = (pred == labs).sum().cpu().detach().numpy() / (pred.shape[0] / 1.0)

            print("Iter: {}/{}, trn_loss: {}, tst_accuracy: {}".format(i, iters, trn_loss_clasifier, tst_accuracy))

    classifier = classifier.cpu()
    T.save(classifier, os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/classifier.p"))
    print("Done")


def _test_mux_rnn_policies(policy_dict, env_list, n_envs):
    env = hex_env.Hexapod(env_list, max_n_envs=n_envs)
    env.env_change_prob = 1
    env.max_steps = env.max_steps
    classifier = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/classifier.p"), map_location='cpu')

    # Test visually
    while True:
        current_idx = 0
        s = env.reset()
        h_p = None
        h_c = None
        episode_reward = 0
        with T.no_grad():
            for i in range(env.max_steps * 2):
                env_idx, h_c = classifier((my_utils.to_tensor(s, True).unsqueeze(0), h_c))
                env_idx = T.argmax(env_idx[0][0]).numpy()
                if env_idx != current_idx:
                    current_idx = env_idx
                    h_p = None
                    print("Changing policy to: {}".format(env_list[env_idx]))

                act, h_p = policy_dict[env_list[env_idx]]((my_utils.to_tensor(s, True).unsqueeze(0), h_p))
                s, r, done, _ = env.step(act[0][0].numpy())
                episode_reward += r
                env.render()
                print("Env classification: {}".format(env_list[env_idx]))
        print("Episode reward: {}".format(episode_reward))


def _test_mux_reactive_policies(policy_dict, env_list):
    env = hex_env.Hexapod(env_list, max_n_envs=len(env_list))
    env.env_change_prob = 1
    env.max_steps = len(env_list) * 200
    classifier = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/classifier.p"), map_location='cpu')

    # Test visually
    while True:
        s = env.reset()
        h_c = None
        episode_reward = 0
        with T.no_grad():
            for i in range(env.max_steps * 2):
                env_idx, h_c = classifier((my_utils.to_tensor(s, True).unsqueeze(0), h_c))
                env_idx = T.argmax(env_idx[0][0]).numpy()
                act = policy_dict[env_list[env_idx]](my_utils.to_tensor(s, True))
                s, r, done, _ = env.step(act[0].numpy())
                episode_reward += r
                env.render()
                print("Env classification: {}".format(env_list[env_idx]))
        print("Episode reward: {}".format(episode_reward))


if __name__=="__main__": # F57 GIW IPI LT3 MEQ
    T.set_num_threads(1)

    expert_tiles = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '../../algos/PG/agents/Hexapod_RNN_V3_LN_PG_W0E_pg.p'))
    expert_holes = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '../../algos/PG/agents/Hexapod_RNN_V3_LN_PG_IZ1_pg.p'))
    expert_pipe = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../algos/PG/agents/Hexapod_RNN_V3_LN_PG_GMV_pg.p'))

    env_list = ["holes", "pipe", "holes", "pipe"]
    expert_dict = {"holes" : expert_holes, "pipe" : expert_pipe}

    if True:
        make_dataset_rnn_experts(env_list=env_list,
                                 expert_dict=expert_dict,
                                 N=3000, n_envs=3, render=False)
    if True:
        train_classifier(n_classes=2, iters=10000, env_list=env_list)
    if False:
        _test_mux_rnn_policies(expert_dict, env_list, n_envs=3)
