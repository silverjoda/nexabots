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



def make_reactive_dataset(env_list, expert_dict, ID, N, n_envs, render=False):
    env = hex_env.Hexapod(env_list)
    env.env_change_prob = 0.
    length = n_envs * 200

    episode_states = []
    episode_labels = []
    episode_acts = []
    ctr = 0
    while ctr < N:

        # Print info
        print("Iter: {}".format(ctr))

        # Generate new environment
        envs, size_list, scaled_indeces_list = env.generate_hybrid_env(n_envs, length)
        scaled_indeces_list.append(length)

        cr = 0
        states = []
        acts = []
        labels = []

        current_env_idx = 0
        current_env = envs[current_env_idx]
        policy = expert_dict[current_env]

        print(envs, scaled_indeces_list, current_env)

        s = env.reset()
        for j in range(n_envs * 400):
            x = env.sim.get_state().qpos.tolist()[0] * 100 + 10

            if x > scaled_indeces_list[current_env_idx]:
                print(x)
                current_env_idx += 1
                current_env = envs[current_env_idx]
                print(current_env)
                policy = expert_dict[current_env]
                print("Policy switched to {} policy".format(envs[current_env_idx]))

            states.append(s)
            labels.append(env_list.index(current_env))
            action = policy((my_utils.to_tensor(s, True)))
            action = action[0].detach().numpy() + np.random.randn(env.act_dim) * 0.1
            acts.append(action)
            s, r, done, od, = env.step(action)
            cr += r

            if render:
                env.render()

        if cr < 200:
            continue
        ctr += 1

        episode_states.append(np.stack(states))
        episode_labels.append(np.stack(labels))
        episode_acts.append(np.stack(acts))

        print("Total episode reward: {}".format(cr))

    np_states = np.stack(episode_states)
    np_labels = np.stack(episode_labels)
    np_acts = np.stack(episode_acts)

    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/states_{}.npy".format(ID)), np_states)
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/labels_{}.npy".format(ID)), np_labels)
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/acts_{}.npy".format(ID)), np_acts)


def make_classif_dataset(env_list, expert_dict, ID, N, n_envs, render=False):
    env = hex_env.Hexapod(env_list)
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
        for j in range(n_envs * 400):
            x = env.sim.get_state().qpos.tolist()[0] * 100 + 20

            if x > scaled_indeces_list[current_env_idx]:
                current_env_idx += 1
                current_env = envs[current_env_idx]
                print(current_env)
                policy = expert_dict[current_env]

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

        # if cr < 10:
        #     continue
        ctr += 1

        episode_states.append(np.stack(states))
        episode_labels.append(np.stack(labels))

        print("Total episode reward: {}".format(cr))

    np_states = np.stack(episode_states)
    np_labels = np.stack(episode_labels)

    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/states_classif_{}.npy".format(ID)), np_states)
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/labels_classif_{}.npy".format(ID)), np_labels)


def classify_multiple(n_classes, iters, env_list):
    env = hex_env.Hexapod(env_list)
    classifier = policies.RNN_CLASSIF_ENV(env, hid_dim=32, memory_dim=32, n_temp=3, n_classes=n_classes, to_gpu=True).cuda()
    optimizer_classifier = T.optim.Adam(classifier.parameters(), lr=2e-4)
    lossfun_classifier = T.nn.CrossEntropyLoss()

    # N x EP_LEN x OBS_DIM
    expert_states = np.load("data/states_classif_A.npy")

    # N x EP_LEN x N_CLASSES
    expert_labels = np.load("data/labels_classif_A.npy")

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

        trn_loss_clasifier = lossfun_classifier(labels_T[:, N_WARMUP_STEPS:].contiguous().view(-1, 3), expert_labels_T[:, N_WARMUP_STEPS:].contiguous().view(-1))
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
            pred = T.argmax(labels_T.contiguous().view(-1, 3), dim=1)
            labs = expert_labels_T.contiguous().view(-1)
            tst_loss_clasifier = (pred == labs).sum().cpu().detach().numpy() / (pred.shape[0] / 1.0)

            print("Iter: {}/{}, trn_loss_classifier: {}, tst_loss_classifier: {}".format(i, iters, trn_loss_clasifier, tst_loss_clasifier))


    classifier = classifier.cpu()
    T.save(classifier, "classifier_A.p")
    print("Done")


def test(env_list):
    env = hex_env.Hexapod(env_list)
    master = T.load("master_A.p", map_location='cpu')
    classifier = T.load("classifier_A.p", map_location='cpu')

    env.env_change_prob = 1.

    # Test visually
    while True:
        s = env.reset()
        h_m = None
        h_c = None
        episode_reward = 0
        with T.no_grad():
            for i in range(env.max_steps * 2):
                act, h_m = master((my_utils.to_tensor(s, True).unsqueeze(0), h_m))
                c, h_c = classifier((my_utils.to_tensor(s, True).unsqueeze(0), h_c))
                s, r, done, _ = env.step(act[0][0].numpy())
                episode_reward += r
                env.render()
                print("Env classification: {}".format(env_list[T.argmax(c[0][0]).numpy()]))
        print("Episode reward: {}".format(episode_reward))


def test_classifier_reactive_policies(policy_dict, env_list):
    env = hex_env.Hexapod(env_list)
    env.env_change_prob = 1
    env.max_steps = 600
    classifier = T.load("classifier_A.p", map_location='cpu')

    # Test visually
    while True:
        current_env = "flat"
        env_idx = np.random.randint(0, 3)
        rnd_idx = np.random.randint(0, 3)
        s = env.reset()
        h_c = None
        episode_reward = 0
        with T.no_grad():
            for i in range(env.max_steps * 2):
                env_idx, h_c = classifier((my_utils.to_tensor(s, True).unsqueeze(0), h_c))
                #print(env_idx)
                env_idx = T.argmax(env_idx[0][0]).numpy()
                if np.random.rand() < 0.01:
                    rnd_idx = np.random.randint(0, 3)

                act = policy_dict[env_list[env_idx]](my_utils.to_tensor(s, True))
                s, r, done, _ = env.step(act[0].numpy())
                episode_reward += r
                env.render()
                print("Env classification: {}".format(env_list[env_idx]))
        print("Episode reward: {}".format(episode_reward))


if __name__=="__main__": # F57 GIW IPI LT3 MEQ
    T.set_num_threads(1)

    # Current experts:
    # Generalization: Novar: QO6, Var: OSM
    # flat: P92, DFE
    # tiles: K4F
    # triangles: LBD
    # Stairs: HOS
    # pipe: 9GV
    # perlin: P92


    reactive_expert_tiles = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '../../algos/PG/agents/Hexapod_NN_PG_K4F_pg.p'))
    reactive_expert_stairs = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '../../algos/PG/agents/Hexapod_NN_PG_HOS_pg.p'))
    reactive_expert_pipe = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../algos/PG/agents/Hexapod_NN_PG_9GV_pg.p'))

    env_list = ["tiles", "stairs", "pipe"]
    expert_dict = {"tiles" : reactive_expert_tiles, "stairs" : reactive_expert_stairs, "pipe" : reactive_expert_pipe}

    if True:
        make_reactive_dataset(env_list=env_list,
                     expert_dict = expert_dict,
                     ID="REACTIVE", N=1000, n_envs=3, render=True)
    if True:
        make_classif_dataset(env_list=env_list,
                              expert_dict=expert_dict,
                              ID="A", N=2000, n_envs=3, render=True)
    if True:
        classify_multiple(n_classes=3, iters=20000, env_list=env_list)
    if True:
        #test(env_list)
        #test_classifier(expert_dict, env_list)
        test_classifier_reactive_policies(expert_dict, env_list)