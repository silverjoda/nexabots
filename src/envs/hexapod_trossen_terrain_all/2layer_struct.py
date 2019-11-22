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


def make_dataset_reactive_experts(env_list, expert_dict, N, n_envs, render=False, ID="def"):
    env = hex_env.Hexapod(env_list, max_n_envs=3, specific_env_len=25, s_len=200)
    env.env_change_prob = 0.0  # THIS HAS TO BE ZERO!!!
    change_prob = 0.01

    forced_episode_length = 160 * n_envs
    env_length = env.specific_env_len * n_envs
    physical_env_len = env.env_scaling * n_envs * 2
    physical_env_offset = physical_env_len * 0.2

    episode_states = []
    episode_labels = []
    ctr = 0
    while ctr < N:

        # Print info
        print("Iter: {}".format(ctr))

        # Generate new environment
        envs, size_list, scaled_indeces_list = env.generate_hybrid_env(n_envs, env_length)
        scaled_indeces_list.append(env.specific_env_len * n_envs)
        raw_indeces_list = [s / float(env_length) for s in scaled_indeces_list]

        cr = 0
        states = []
        labels = []

        bad_episode = False

        current_env_idx = 0
        current_env = envs[current_env_idx]
        policy = expert_dict[current_env]

        s = env.reset()
        for j in range(forced_episode_length):
            x = env.sim.get_state().qpos.tolist()[0]

            if x > raw_indeces_list[current_env_idx] * physical_env_len + 0.05 - physical_env_offset:
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
            s, r, done, (vr, dr) = env.step(action)
            cr += r

            if render:
                env.render()

            if done and j < env.max_steps - 1:
                bad_episode = True

        if bad_episode:
            print("Discarded episode")
            continue
        ctr += 1

        episode_states.append(np.stack(states))
        episode_labels.append(np.stack(labels))

        print("Total episode reward: {}".format(cr))

    np_states = np.stack(episode_states)
    np_labels = np.stack(episode_labels)

    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/states_{}.npy".format(ID)), np_states)
    np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/labels_{}.npy".format(ID)), np_labels)




def train_classifier(n_classes, iters, env_list, ID="def"):
    env = hex_env.Hexapod(env_list, max_n_envs=3, specific_env_len=25, s_len=200)
    classifier = policies.RNN_CLASSIF_BASIC(env, hid_dim=48, memory_dim=48, n_temp=3, n_classes=n_classes, to_gpu=True)

    if T.cuda.is_available():
        classifier = classifier.cuda()

    optimizer_classifier = T.optim.Adam(classifier.parameters(), lr=2e-4, weight_decay=0.001)
    lossfun_classifier = T.nn.CrossEntropyLoss()

    # N x EP_LEN x OBS_DIM
    expert_states = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/states_{}.npy".format(ID)))

    # N x EP_LEN x N_CLASSES
    expert_labels = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/labels_{}.npy".format(ID)))

    batchsize = 48

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

        batch_states_T = T.from_numpy(batch_states).float()
        expert_labels_T = T.from_numpy(batch_labels).long()

        if T.cuda.is_available():
            batch_states_T = batch_states_T.cuda()
            expert_labels_T = expert_labels_T.cuda()

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
            batch_states_T = T.from_numpy(states).float()
            expert_labels_T = T.from_numpy(labels).long()

            if T.cuda.is_available():
                batch_states_T = batch_states_T.cuda()
                expert_labels_T = expert_labels_T.cuda()

            labels_T, _ = classifier.forward((batch_states_T, None))
            pred = T.argmax(labels_T.contiguous().view(-1, n_classes), dim=1)
            labs = expert_labels_T.contiguous().view(-1)
            tst_accuracy = (pred == labs).sum().cpu().detach().numpy() / (pred.shape[0] / 1.0)

            print("Iter: {}/{}, trn_loss: {}, tst_accuracy: {}".format(i, iters, trn_loss_clasifier, tst_accuracy))

    classifier = classifier.cpu()
    T.save(classifier, os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/classifier_{}.p".format(ID)))
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


def _test_mux_reactive_policies(policy_dict, env_list, n_envs, ID='def'):
    import cv2

    def printval(values):
        img = np.zeros((90, 200, 3), dtype=np.uint8)
        a_idx = np.argmax(values)
        cv2.putText(img, 'p_{}'.format(env_list[0]) + '{0:.2f}'.format(values[0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 * int(a_idx != 0), 255, 0),
                    1, cv2.LINE_AA)
        cv2.putText(img, 'p_{}'.format(env_list[1])  + '{0:.2f}'.format(values[1]), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 * int(a_idx != 1), 255, 0),
                    1, cv2.LINE_AA)
        cv2.putText(img, 'p_{}'.format(env_list[2])  + '{0:.2f}'.format(values[2]), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 * int(a_idx != 2), 255, 0),
                    1, cv2.LINE_AA)
        cv2.imshow('classification', img)
        cv2.waitKey(1)

    env = hex_env.Hexapod(env_list, max_n_envs=3, specific_env_len=25, s_len=200, walls=False)
    env.env_change_prob = 1
    classifier = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/classifier_{}.p".format(ID)), map_location='cpu')

    # Test visually
    while True:
        s = env.reset()
        h_c = None
        episode_reward = 0
        with T.no_grad():
            for i in range(env.max_steps + 400):
                env_dist, h_c = classifier((my_utils.to_tensor(s, True).unsqueeze(0), h_c))
                env_softmax = T.softmax(env_dist, 2)[0][0].numpy()
                env_idx = T.argmax(env_dist[0][0]).numpy()
                printval(env_softmax)

                act = policy_dict[env_list[env_idx]](my_utils.to_tensor(s, True))

                s, r, done, _ = env.step(act[0].numpy())
                episode_reward += r
                env.render()
                #print("Env classification: {}".format(env_list[env_idx]))
        print("Episode reward: {}".format(episode_reward))


def _test_full_comparison(expert_dict, env_list, n_envs, N, render=False, ID='def'):
    env = hex_env.Hexapod(env_list, max_n_envs=3, specific_env_len=25, s_len=200, walls=False)
    env.env_change_prob = 0
    env.rnd_yaw = False
    classifier = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/classifier_{}.p".format(ID)), map_location='cpu')

    forced_episode_length = 700
    env_length = env.specific_env_len * n_envs
    physical_env_len = env.env_scaling * n_envs * 2
    physical_env_offset = physical_env_len * 0.15

    success_ctr = 0

    print("Testing Oracle expert selection")
    env.setseed(1337)
    cr = 0
    cdr = 0
    for _ in range(0):
        # Generate new environment
        envs, size_list, scaled_indeces_list = env.generate_hybrid_env(n_envs, env_length)
        scaled_indeces_list.append(env.specific_env_len * n_envs)
        raw_indeces_list = [s / float(env_length) for s in scaled_indeces_list]

        dr = 0

        current_env_idx = 0
        current_env = envs[current_env_idx]
        policy = expert_dict[current_env]

        s = env.reset()
        bad_episode = False
        for j in range(forced_episode_length):
            x = env.sim.get_state().qpos.tolist()[0]

            if x > raw_indeces_list[current_env_idx] * physical_env_len + 0.05 - physical_env_offset:
                current_env_idx += 1
                current_env = envs[current_env_idx]
                policy = expert_dict[envs[current_env_idx]]
                print("Env switched to {} ".format(envs[current_env_idx]))

            # print(current_env)
            action = policy((my_utils.to_tensor(s, True)))
            action = action[0].detach().numpy()
            s, r, done, (_, d_r) = env.step(action)
            cr += r
            dr = max(dr, d_r)

            if render:
                env.render()

            if done and j < env.max_steps - 1:
                bad_episode = True


        if not bad_episode:
            success_ctr += 1


        cdr += dr

    cum_gait_quality_oracle = cr / float(N)
    cum_dist_quality_oracle = cdr / float(N)
    success_rate_oracle = success_ctr / float(N)

    # ===================================================================================
    # ===================================================================================
    print("Testing MUX expert selection")
    env.setseed(1337)
    cr = 0
    cdr = 0
    success_ctr = 0
    for _ in range(0):
        # Generate new environment
        envs, size_list, scaled_indeces_list = env.generate_hybrid_env(n_envs, env_length)
        scaled_indeces_list.append(env.specific_env_len * n_envs)

        dr = 0

        s = env.reset()
        h_c = None

        bad_episode = False
        with T.no_grad():
            for j in range(forced_episode_length):
                env_dist, h_c = classifier((my_utils.to_tensor(s, True).unsqueeze(0), h_c))
                env_idx = T.argmax(env_dist[0][0]).numpy()

                # print(current_env)
                act = expert_dict[env_list[env_idx]](my_utils.to_tensor(s, True))

                s, r, done, (_, d_r) = env.step(act[0].numpy())
                cr += r
                dr = max(dr, d_r)

                if render:
                    env.render()

                if done and j < env.max_steps - 1:
                    bad_episode = True


        if not bad_episode:
            success_ctr += 1

        cdr += dr

    cum_gait_quality_MUX = cr / float(N)
    cum_dist_quality_MUX = cdr / float(N)
    success_rate_MUX = success_ctr / float(N)


    # ===================================================================================
    # ===================================================================================
    print("Testing RNN")
    env.setseed(int(time.time()))
    cr = 0
    cdr = 0
    success_ctr = 0
    for _ in range(N):
        # Generate new environment
        _ = env.generate_hybrid_env(n_envs, env_length)

        dr = 0

        policy = expert_dict["rnn"]

        s = env.reset()
        h = None
        bad_episode = False
        for j in range(forced_episode_length):
            action, h = policy((my_utils.to_tensor(s, True).unsqueeze(0), h))
            action = action[0].detach().numpy()
            s, r, done, (_, d_r) = env.step(action)
            cr += r
            dr = max(dr, d_r)

            if render:
                env.render()

            if done and j < env.max_steps - 1:
                bad_episode = True

        if not bad_episode:
            success_ctr += 1

        cdr += dr

    cum_gait_quality_RNN = cr / float(N)
    cum_dist_quality_RNN = cdr / float(N)
    success_rate_RNN = success_ctr / float(N)

    print("RESULTS: ")
    print("Oracle MUX -> Average gait quality: {}, Average maximum reached distance: {}, success rate: {}".format(cum_gait_quality_oracle,
                                                                                            cum_dist_quality_oracle, success_rate_oracle))
    print("classifier MUX -> Average gait quality: {}, Average maximum reached distance: {}, success rate: {}".format(cum_gait_quality_MUX,
                                                                                            cum_dist_quality_MUX, success_rate_MUX))
    print("RNN -> Average gait quality: {}, Average reached distance: {}, success rate: {}".format(cum_gait_quality_RNN,
                                                                                            cum_dist_quality_RNN,
                                                                                                           success_rate_RNN))



if __name__=="__main__": # F57 GIW IPI LT3 MEQ
    T.set_num_threads(1)

    reactive_expert_tiles = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                '../../algos/PG/agents/Hexapod_NN_PG_UAN_pg.p'))
    reactive_expert_stairs = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 '../../algos/PG/agents/Hexapod_NN_PG_3LM_pg.p'))
    reactive_expert_pipe = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '../../algos/PG/agents/Hexapod_NN_PG_CED_pg.p'))

    reactive_expert_triangles = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '../../algos/PG/agents/Hexapod_NN_PG_NEZ_pg.p'))
    reactive_expert_flat = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '../../algos/PG/agents/Hexapod_NN_PG_O4W_pg.p'))

    rnn_expert = T.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               '../../algos/PG/agents/Hexapod_RNN_PG_OJB_pg.p')) #OJB

    env_list = ["triangles", "flat", "perlin"]
    expert_dict = {"tiles": reactive_expert_tiles,
                   "pipe": reactive_expert_pipe,
                   "stairs": reactive_expert_stairs,
                   "triangles": reactive_expert_triangles,
                   "flat": reactive_expert_flat,
                   "rnn": rnn_expert}

    if False:
        make_dataset_reactive_experts(env_list=env_list,
                                 expert_dict=expert_dict,
                                 N=20000, n_envs=3, render=False, ID="FINAL")
    if False:
        train_classifier(n_classes=3, iters=20000, env_list=env_list, ID="FINAL")
    if True:
        _test_mux_reactive_policies(expert_dict, env_list, n_envs=3, ID="FINAL")
    if False:
        _test_full_comparison(expert_dict, env_list, N=100, n_envs=3, render=True, ID="FINAL")

