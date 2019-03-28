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


def make_dataset(env_list, expert_dict, ID, N, n_envs, render=False):
    env = hex_env.Hexapod(env_list)
    length = n_envs * 200

    h = None

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
            x = env.sim.get_state().qpos.tolist()[0] * 100 + 20

            if x > scaled_indeces_list[current_env_idx]:
                print(x)
                current_env_idx += 1
                current_env = envs[current_env_idx]
                print(current_env)
                policy = expert_dict[current_env]
                h = None
                print("Policy switched to {} policy".format(envs[current_env_idx]))

            states.append(s)
            labels.append(env_list.index(current_env))
            action, h = policy((my_utils.to_tensor(s, True).unsqueeze(0), h))
            action = action[0][0].detach().numpy()
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


def make_reactive_dataset(env_list, expert_dict, ID, N, n_envs, render=False):
    env = hex_env.Hexapod(env_list)
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
            action = action[0].detach().numpy()
            acts.append(action)
            s, r, done, od, = env.step(action)
            cr += r

            if render:
                env.render()

        if cr < 100:
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

            if np.random.rand() < change_prob:
                print(x)
                current_env_idx = np.random.randint(0, len(env_list))
                current_env = envs[current_env_idx]
                print(current_env)
                policy = expert_dict[current_env]
                print("Policy switched to {} policy".format(envs[current_env_idx]))

            states.append(s)
            labels.append(env_list.index(current_env))
            action = policy((my_utils.to_tensor(s, True)))
            action = action[0].detach().numpy()
            s, r, done, od, = env.step(action)
            cr += r

            if render:
                env.render()

        if cr < 200:
            continue
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


def imitate_multiple(n_classes):
    env = hex_env.Hexapod()
    master = policies.RNN_V3_PG(env, hid_dim=64, memory_dim=32, n_temp=3, tanh=True, to_gpu=True).cuda()
    classifier = policies.RNN_CLASSIF_ENV(env, hid_dim=32, memory_dim=32, n_temp=3, n_classes=n_classes, to_gpu=True).cuda()
    optimizer_master = T.optim.Adam(master.parameters(), lr=2e-4)
    #optimizer_classifier = T.optim.Adam(classifier.parameters(), lr=3e-4)
    lossfun_master = T.nn.MSELoss()
    #lossfun_classifier = T.nn.CrossEntropyLoss()

    # N x EP_LEN x OBS_DIM
    expert_states = np.load("data/states_A.npy")
    # N x EP_LEN x ACT_DIM
    expert_acts = np.load("data/acts_A.npy")
    # N x EP_LEN x N_CLASSES
    #expert_labels = np.load("data/labels_A.npy")

    iters = 20000
    batchsize = 32

    assert len(expert_states) == len(expert_acts)
    N_EPS, EP_LEN, OBS_DIM = expert_states.shape
    _, _, ACT_DIM = expert_acts.shape

    for i in range(iters):
        # Make batch of episodes
        batch_states = []
        batch_acts = []
        #batch_labels = []
        for _ in range(batchsize):
            rnd_idx = np.random.randint(0, N_EPS)
            states = expert_states[rnd_idx]
            acts = expert_acts[rnd_idx]
            #labels = expert_labels[rnd_idx]
            batch_states.append(states)
            batch_acts.append(acts)
            #batch_labels.append(labels)

        batch_states = np.stack(batch_states)
        batch_acts = np.stack(batch_acts)
        #batch_labels = np.stack(batch_labels)

        assert batch_states.shape == (batchsize, EP_LEN, OBS_DIM)
        assert batch_acts.shape == (batchsize, EP_LEN, ACT_DIM)

        batch_states_T = T.from_numpy(batch_states).float().cuda()
        expert_acts_T = T.from_numpy(batch_acts).float().cuda()
        #expert_labels_T = T.from_numpy(batch_labels).long().cuda()

        # Perform batch forward pass on episodes
        master_acts_T, _ = master.forward((batch_states_T, None))
        master_labels_T, _ = classifier.forward((batch_states_T, None))

        # Update RNN
        N_WARMUP_STEPS = 0

        # loss_clasifier = lossfun_classifier(master_labels_T[:, N_WARMUP_STEPS:].contiguous().view(-1, 4), expert_labels_T[:, N_WARMUP_STEPS:].contiguous().view(-1))
        # loss_clasifier.backward()
        # classifier.soft_clip_grads()
        # optimizer_classifier.step()

        loss_master = lossfun_master(master_acts_T[:, N_WARMUP_STEPS:, :], expert_acts_T[:, N_WARMUP_STEPS:, :])
        loss_master.backward()
        master.soft_clip_grads()
        optimizer_master.step()

        # Print info
        if i % 10 == 0:
            print("Iter: {}/{}, loss_master: {}, loss_classifier: {}".format(i, iters, loss_master, None))

    master = master.cpu()
    classifier = classifier.cpu()
    T.save(master, "master_A.p")
    T.save(classifier, "classifier_A.p")
    print("Done")


def classify_multiple(n_classes):
    env = hex_env.Hexapod()
    classifier = policies.RNN_CLASSIF_ENV(env, hid_dim=32, memory_dim=32, n_temp=3, n_classes=n_classes, to_gpu=True).cuda()
    optimizer_classifier = T.optim.Adam(classifier.parameters(), lr=3e-4)
    lossfun_classifier = T.nn.CrossEntropyLoss()

    # N x EP_LEN x OBS_DIM
    expert_states = np.load("data/states_A.npy")

    # N x EP_LEN x N_CLASSES
    expert_labels = np.load("data/labels_A.npy")

    iters = 10000
    batchsize = 32

    N_EPS, EP_LEN, OBS_DIM = expert_states.shape


    for i in range(iters):
        # Make batch of episodes
        batch_states = []
        batch_labels = []
        for _ in range(batchsize):
            rnd_idx = np.random.randint(0, N_EPS)
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

        loss_clasifier = lossfun_classifier(labels_T[:, N_WARMUP_STEPS:].contiguous().view(-1, 4), expert_labels_T[:, N_WARMUP_STEPS:].contiguous().view(-1))
        loss_clasifier.backward()
        classifier.soft_clip_grads()
        optimizer_classifier.step()

        # Print info
        if i % 10 == 0:
            print("Iter: {}/{}, loss_classifier: {}".format(i, iters, None))

    classifier = classifier.cpu()
    T.save(classifier, "classifier_A.p")
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


def test_classifier(policy_dict, env_list):
    env = hex_env.Hexapod(env_list)
    classifier = T.load("classifier_A.p", map_location='cpu')

    # Test visually
    while True:
        current_env = "flat"
        s = env.reset()
        h_p = None
        h_c = None
        episode_reward = 0
        with T.no_grad():
            for i in range(env.max_steps * 2):
                env_idx, h_c = classifier((my_utils.to_tensor(s, True).unsqueeze(0), h_c))
                print(env_idx)
                env_idx = T.argmax(env_idx[0][0]).numpy()
                if env_list[env_idx] != current_env:
                    current_env = env_list[env_idx]
                    h_p = None

                act, h_p = policy_dict[current_env]((my_utils.to_tensor(s, True).unsqueeze(0), h_p))
                s, r, done, _ = env.step(act[0][0].numpy())
                episode_reward += r
                env.render()
                #print("Env classification: {}".format(env_list[env_idx]))
        print("Episode reward: {}".format(episode_reward))


def test_classifier_reactive_policies(policy_dict, env_list):
    env = hex_env.Hexapod(env_list)
    classifier = T.load("classifier_A.p", map_location='cpu')

    # Test visually
    while True:
        current_env = "flat"
        s = env.reset()
        h_c = None
        episode_reward = 0
        with T.no_grad():
            for i in range(env.max_steps * 2):
                env_idx, h_c = classifier((my_utils.to_tensor(s, True).unsqueeze(0), h_c))
                print(env_idx)
                env_idx = T.argmax(env_idx[0][0]).numpy()
                if env_list[env_idx] != current_env:
                    current_env = env_list[env_idx]

                act = policy_dict[current_env](my_utils.to_tensor(s, True))
                s, r, done, _ = env.step(act[0].numpy())
                episode_reward += r
                env.render()
                #print("Env classification: {}".format(env_list[env_idx]))
        print("Episode reward: {}".format(episode_reward))


if __name__=="__main__": # F57 GIW IPI LT3 MEQ
    T.set_num_threads(1)
    expert_flat = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),                                                            
                         '../../algos/PG/agents/Hexapod_RNN_V3_PG_FEK_pg.p'))
    expert_tiles = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../algos/PG/agents/Hexapod_RNN_V3_PG_6GY_pg.p'))
    expert_holes = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../algos/PG/agents/Hexapod_RNN_V3_PG_I51_pg.p'))
    expert_pipe = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../algos/PG/agents/Hexapod_RNN_V3_PG_8DF_pg.p'))
    # expert_inversholes = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                                   '../../algos/PG/agents/Hexapod_RNN_V3_PG_MEQ_pg.p'))

    reactive_expert_flat = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../algos/PG/agents/Hexapod_NN_PG_K55_pg.p'))
    reactive_expert_tiles = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '../../algos/PG/agents/Hexapod_NN_PG_P4D_pg.p'))
    reactive_expert_holes = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '../../algos/PG/agents/Hexapod_NN_PG_OEO_pg.p'))
    reactive_expert_pipe = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../algos/PG/agents/Hexapod_NN_PG_HIS_pg.p'))
    reactive_expert_inverseholes = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '../../algos/PG/agents/Hexapod_NN_PG_K9B_pg.p'))

    env_list = ["flat", "holes", "pipe"]
    expert_dict = {"flat" : reactive_expert_flat, "tiles" : reactive_expert_tiles, "holes" : reactive_expert_holes, "pipe" : reactive_expert_pipe}
    # if False:
    #     make_dataset(env_list=env_list,
    #                  expert_dict = expert_dict,
    #                  ID="A", N=1000, n_envs=3, render=False)
    if False:
        make_reactive_dataset(env_list=env_list,
                     expert_dict = expert_dict,
                     ID="REACTIVE", N=1000, n_envs=3, render=False)
    if False:
        make_classif_dataset(env_list=env_list,
                              expert_dict=expert_dict,
                              ID="A", N=1000, n_envs=3, render=False)
    if False:
        imitate_multiple(n_classes=4)
    if False:
        classify_multiple(n_classes=4)
    if False:
        #test(env_list)
        #test_classifier(expert_dict, env_list)
        test_classifier_reactive_policies(expert_dict, env_list)
