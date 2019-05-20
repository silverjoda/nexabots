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
    episode_length = n_envs * 200
    env.env_change_prob = 0.0  # THIS HAS TO BE ZERO!!!
    change_prob = 0.01
    env.s_len = 130
    env.max_steps = env.n_envs * env.s_len

    h = None

    episode_states = []
    episode_labels = []
    ctr = 0
    while ctr < N:

        # Print info
        print("Iter: {}".format(ctr))

        # Generate new environment
        envs, size_list, scaled_indeces_list = env.generate_hybrid_env(n_envs, episode_length)
        scaled_indeces_list.append(episode_length)

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

        if cr < 0:
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
    env.s_len = 130
    env.max_steps = env.n_envs * env.s_len

    change_prob = 0.01
    env.env_change_prob = 0.0 # THIS HAS TO BE ZERO!!!

    episode_states = []
    episode_labels = []
    ctr = 0
    while ctr < N:

        bad_episode = False

        # Print info
        print("Iter: {}".format(ctr))

        # Generate new environment
        envs, size_list, scaled_indeces_list = env.generate_hybrid_env(n_envs, env.max_steps)
        scaled_indeces_list.append(env.max_steps)

        cr = 0
        states = []
        labels = []

        current_env_idx = 0
        current_env = envs[current_env_idx]
        policy = expert_dict[current_env]

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
                #print("Policy switched to {} policy".format(policy_choice))

            states.append(s)
            labels.append(env_list.index(current_env))
            #print(current_env)
            action = policy((my_utils.to_tensor(s, True)))
            action = action[0].detach().numpy()
            s, r, done, od, = env.step(action)
            cr += r

            if render:
                env.render()

        if cr < 0:
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


def train_classifier_iteratively(env_list, expert_dict, iters, n_envs, n_classes, render=False):
    env = hex_env.Hexapod(env_list, max_n_envs=n_envs)
    episode_length = n_envs * 200
    env.env_change_prob = 0.0 # THIS HAS TO BE ZERO!!!
    env.s_len = 130
    env.max_steps = env.n_envs * env.s_len
    batchsize = 24

    classifier = policies.RNN_CLASSIF_ENV(env, hid_dim=24, memory_dim=24, n_temp=2, n_classes=n_classes,
                                          to_gpu=True).cuda()
    optimizer_classifier = T.optim.Adam(classifier.parameters(), lr=2e-4, weight_decay=0.001)
    lossfun_classifier = T.nn.CrossEntropyLoss()

    for i in range(iters):
        states_batch = []
        labels_batch = []
        for b in range(batchsize):

            # Generate new environment
            envs, size_list, scaled_indeces_list = env.generate_hybrid_env(n_envs, env.max_steps)
            scaled_indeces_list.append(env.max_steps)

            cr = 0
            states = []
            labels = []

            current_env_idx = 0
            current_env = envs[current_env_idx]
            #print(envs, scaled_indeces_list, current_env)

            s = env.reset()
            h_c = None
            with T.no_grad():
                for e in range(episode_length):
                    x = env.sim.get_state().qpos.tolist()[0] * 100 + 20

                    if x > scaled_indeces_list[current_env_idx]:
                        current_env_idx += 1
                        current_env = envs[current_env_idx]
                        #print("Env switched to {} ".format(envs[current_env_idx]))

                    env_idx, h_c = classifier((my_utils.to_tensor(s, True).unsqueeze(0).cuda(), h_c))
                    env_idx = T.argmax(env_idx[0][0]).cpu().numpy()

                    policy = expert_dict[envs[env_idx]]

                    states.append(s)
                    labels.append(env_list.index(current_env))
                    action = policy((my_utils.to_tensor(s, True)))
                    action = action[0].cpu().numpy()
                    s, r, done, od, = env.step(action)
                    cr += r

                    if render:
                        env.render()

            states_batch.append(np.stack(states))
            labels_batch.append(np.stack(labels))

        np_states = np.stack(states_batch)
        np_labels = np.stack(labels_batch)

        batch_states_T = T.from_numpy(np_states).float().cuda()
        expert_labels_T = T.from_numpy(np_labels).long().cuda()

        # Perform batch forward pass on episodes
        labels_T, _ = classifier.forward((batch_states_T, None))

        # Update RNN
        trn_loss_clasifier = lossfun_classifier(labels_T[:, :].contiguous().view(-1, n_classes),
                                                expert_labels_T[:, :].contiguous().view(-1))
        trn_loss_clasifier.backward()
        classifier.soft_clip_grads()
        optimizer_classifier.step()

        if i % 5 == 0:
            pred = T.argmax(labels_T.contiguous().view(-1, n_classes), dim=1)
            labs = expert_labels_T.contiguous().view(-1)
            trn_accuracy = (pred == labs).sum().cpu().detach().numpy() / (pred.shape[0] / 1.0)
            print("Iter: {}/{}, trn_loss: {}, trn_accuracy: {}".format(i, iters, trn_loss_clasifier, trn_accuracy))


    classifier = classifier.cpu()
    T.save(classifier, os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/classifier_iter.p"))
    print("Done")


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


def _test_mux_reactive_policies(policy_dict, env_list, n_envs):
    import cv2

    def printval(values):
        img = np.zeros((90, 200, 3), dtype=np.uint8)
        a_idx = np.argmax(values)
        cv2.putText(img, 'p_holes = {0:.2f}'.format(values[0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 * int(a_idx != 0), 255, 0),
                    1, cv2.LINE_AA)
        cv2.putText(img, 'p_pipe = {0:.2f}'.format(values[1]), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 * int(a_idx != 1), 255, 0),
                    1, cv2.LINE_AA)
        cv2.putText(img, 'p_tiles = {0:.2f}'.format(values[2]), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 * int(a_idx != 2), 255, 0),
                    1, cv2.LINE_AA)
        cv2.imshow('classification', img)
        cv2.waitKey(10)

    env = hex_env.Hexapod(env_list, max_n_envs=n_envs)
    env.env_change_prob = 1
    env.s_len = 130
    env.max_steps = env.s_len * n_envs
    classifier = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/classifier.p"), map_location='cpu')

    # Test visually
    while True:
        s = env.reset()
        h_c = None
        episode_reward = 0
        with T.no_grad():
            for i in range(env.max_steps + 200):

                env_dist, h_c = classifier((my_utils.to_tensor(s, True).unsqueeze(0), h_c))
                env_softmax = T.softmax(env_dist, 2)[0][0].numpy()
                env_idx = T.argmax(env_dist[0][0]).numpy()
                printval(env_softmax)

                act = policy_dict[env_list[env_idx]](my_utils.to_tensor(s, True))

                s, r, done, _ = env.step(act[0].numpy())
                episode_reward += r
                env.render()
                print("Env classification: {}".format(env_list[env_idx]))
        print("Episode reward: {}".format(episode_reward))



def _test_mux_reactive_policies_debug(policy_dict, env_list, n_envs):

    env = hex_env.Hexapod(env_list, max_n_envs=n_envs)
    env.env_change_prob = 1
    env.max_steps = env.max_steps
    classifier = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/classifier.p"), map_location='cpu')

    # Remove this later
    policy_choice = np.random.choice(env_list, 1)[0] #
    policy = policy_dict[policy_choice] #
    # N x EP_LEN x OBS_DIM
    expert_states = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/states.npy"))

    # N x EP_LEN x N_CLASSES
    expert_labels = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/labels.npy"))

    batchsize = 32

    N_EPS, EP_LEN, OBS_DIM = expert_states.shape

    states = expert_states[-32:]
    labels = expert_labels[-32:]
    batch_states_T = T.from_numpy(states).float()
    expert_labels_T = T.from_numpy(labels).long()
    labels_T, _ = classifier.forward((batch_states_T, None))
    pred = T.argmax(labels_T.contiguous().view(-1, 3), dim=1)
    labs = expert_labels_T.contiguous().view(-1)
    tst_accuracy = (pred == labs).sum().cpu().detach().numpy() / (pred.shape[0] / 1.0)

    print("Tst_accuracy: {}".format(tst_accuracy))

    # Test visually
    while True:
        s = env.reset()
        h_c = None
        episode_reward = 0

        states = []
        labels = []

        with T.no_grad():
            for i in range(env.max_steps):

                # Remove this later
                if np.random.rand() < 0.01:
                    policy_choice = np.random.choice(env_list, 1)[0]
                    policy = policy_dict[policy_choice]
                    print("Policy switched to {} policy".format(policy_choice))

                env_idx, h_c = classifier((my_utils.to_tensor(s, True).unsqueeze(0), h_c))
                env_idx = T.argmax(env_idx[0][0]).numpy()

                # Change this later
                act = policy_dict[env_list[env_idx]](my_utils.to_tensor(s, True))

                s, r, done, _ = env.step(act[0].numpy())
                episode_reward += r
                env.render()
                print("Env classification: {}".format(env_list[env_idx]))
        print("Episode reward: {}".format(episode_reward))


if __name__=="__main__": # F57 GIW IPI LT3 MEQ
    T.set_num_threads(1)

    expert_tiles = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '../../algos/PG/agents/Hexapod_NN_PG_3Z0_pg.p'))
    expert_holes = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '../../algos/PG/agents/Hexapod_NN_PG_KE1_pg.p'))
    expert_pipe = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../algos/PG/agents/Hexapod_NN_PG_WSJ_pg.p'))

    env_list = ["holes", "pipe", "tiles"]
    expert_dict = {"holes" : expert_holes, "pipe" : expert_pipe, "tiles" : expert_tiles}

    if False:
        train_classifier_iteratively(env_list, expert_dict, iters=600, n_envs=3, n_classes=3, render=False)
    if False:
        make_dataset_reactive_experts(env_list=env_list,
                                 expert_dict=expert_dict,
                                 N=1500, n_envs=3, render=False)
    if False:
        train_classifier(n_classes=3, iters=15000, env_list=env_list)
    if True:
        _test_mux_reactive_policies(expert_dict, env_list, n_envs=3)
