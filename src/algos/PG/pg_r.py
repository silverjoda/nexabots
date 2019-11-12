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

def train(env, policy, params):

    policy_optim = T.optim.Adam(policy.parameters(), lr=params["lr"], weight_decay=params["decay"], eps=1e-4)

    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_terminals = []

    episode_ctr = 0
    episode_rew = 0

    for i in range(params["iters"]):
        s_0 = env.reset()
        h_0 = None
        done = False
        step_ctr = 0

        # Episode lists
        episode_states = []
        episode_actions = []

        while not done:
            with T.no_grad():
                # Sample action from policy
                action, h_1 = policy.sample_action((my_utils.to_tensor(s_0, True).unsqueeze(0), h_0))

            #print(action.squeeze(0).numpy())

            # Step action
            s_1, r, done, _ = env.step(action.squeeze(0).numpy())
            r = np.clip(r, -5, 5)

            step_ctr += 1
            episode_rew += r

            if params["animate"]:
                env.render()

            # Record transition
            episode_states.append(my_utils.to_tensor(s_0, True))
            episode_actions.append(action)
            batch_rewards.append(my_utils.to_tensor(np.asarray(r, dtype=np.float32), True))
            batch_terminals.append(done)

            s_0 = s_1
            h_0 = h_1

        # Just completed an episode
        episode_ctr += 1

        batch_states.append(T.cat(episode_states))
        batch_actions.append(T.cat(episode_actions))

        # If enough data gathered, then perform update
        if episode_ctr == params["batchsize"]:

            batch_states = T.stack(batch_states)
            batch_actions = T.stack(batch_actions)
            batch_rewards = T.cat(batch_rewards)

            # Scale rewards
            batch_rewards = (batch_rewards - batch_rewards.mean()) / batch_rewards.std()

            # Calculate episode advantages
            batch_advantages = calc_advantages_MC(params["gamma"], batch_rewards, batch_terminals)

            if params["ppo"]:
                update_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, params["ppo_update_iters"])
            else:
                update_proper(policy, policy_optim, batch_states, batch_actions, batch_advantages)

            print("Episode {}/{}, loss_V: {}, loss_policy: {}, mean ep_rew: {}, std: {}".
                  format(i, params["iters"], None, None, episode_rew / params["batchsize"], 1)) # T.exp(policy.log_std).detach().numpy())

            # Finally reset all batch lists
            episode_ctr = 0
            episode_rew = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_terminals = []


        if i % 500 == 0 and i > 0:
            sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "agents/{}_{}_{}_pg.p".format(env.__class__.__name__, policy.__class__.__name__, params["ID"]))
            T.save(policy, sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, params))


        if i % 500 == 0 and i > 0:
            print("Wrote score to file")
            c_score, v_score, d_score = env.test_recurrent(policy, N=100, seed=1337, render=False)
            with open("eval/{}_RNN.txt".format(params["ID"]), "a+") as f:
                f.write("{}, {}, {}, {} \n".format(episode_rew / params["batchsize"], c_score, v_score, d_score))


def update_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, update_iters):
    # Call logprobs on hidden states
    log_probs_old = policy.log_probs(batch_states, batch_actions).detach()
    c_eps = .2

    # Do ppo_update
    for k in range(update_iters):
        log_probs_new = policy.log_probs(batch_states, batch_actions)
        r = T.exp(log_probs_new - log_probs_old).view((-1, 1))
        loss = -T.mean(T.min(r * batch_advantages, r.clamp(1 - c_eps, 1 + c_eps) * batch_advantages))
        policy_optim.zero_grad()
        loss.backward()

        # Step policy update
        policy.soft_clip_grads(3.)
        policy_optim.step()


def update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages):

    # Get action log probabilities
    log_probs = policy.log_probs(batch_states, batch_actions)

    # Calculate loss function
    loss = -T.mean(log_probs.view((-1, 1)) * batch_advantages)

    # Backward pass on policy
    policy_optim.zero_grad()
    loss.backward()

    # Step policy update
    #policy.print_info()
    policy.soft_clip_grads(3.)
    policy_optim.step()

    return loss.data


def update_proper(policy, policy_optim, batch_states, batch_actions, batch_advantages_flat):
    exit()

    # params:
    h_learn = 100
    rollout_step = 10
    h_trunc = 50
    horizon = batch_states.shape[1]
    batchsize = batch_states.shape[0]

    batch_advantages_shaped = batch_advantages_flat.view((batch_states.shape[0], batch_states.shape[1], 1))

    # Do initial rollout to get hidden states
    with T.no_grad():
        hiddens = policy.forward_hidden((batch_states, None))

    # Zero out gradient before starting updates
    policy_optim.zero_grad()

    for i in range(0, horizon):
        start_index = np.maximum(0, i - h_learn)
        h = hiddens[start_index]

        # Get action log probabilities
        log_probs = policy.log_probs_wh(h, batch_states[:, start_index:i+1], batch_actions[:, start_index:i+1])[:,-1:, :]

        # Calculate loss function
        loss = -T.mean(log_probs.view((-1, 1)) * batch_advantages_shaped[:, i:i+1, :].reshape((-1, 1)))

        # Backward pass on policy
        loss.backward()

    for p in policy.parameters():
        assert p.grad is not None
        p.grad = p.grad / batchsize

    if False:
        # Get action log probabilities
        log_probs = policy.log_probs(batch_states, batch_actions)

        # Calculate loss function
        loss = -T.mean(log_probs.view((-1, 1)) * batch_advantages_flat)

        # Backward pass on policy
        policy_optim.zero_grad()
        loss.backward()

    # Step policy update
    policy.soft_clip_grads(5.)
    policy_optim.step()

    return loss.data


def calc_advantages_MC(gamma, batch_rewards, batch_terminals):
    N = len(batch_rewards)

    # Monte carlo estimate of targets
    targets = []
    for i in range(N):
        cumrew = T.tensor(0.)
        for j in range(i, N):
            cumrew += (gamma ** (j - i)) * batch_rewards[j]
            if batch_terminals[j]:
                break
        targets.append(cumrew.view(1, 1))
    targets = T.cat(targets)

    return targets


if __name__=="__main__":
    T.set_num_threads(1)

    env_list = ["tiles", "triangles", "flat"] # 177, 102, 72, -20

    if len(sys.argv) > 1:
        env_list = [sys.argv[1]]

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 1000000, "batchsize": 40, "gamma": 0.995, "lr": 0.001, "decay" : 0.0001, "ppo": True,
              "tanh" : False, "ppo_update_iters": 6, "animate": True, "train" : False,
              "comments" : "Training on 3 envs, /w rep, /w rnd yaw", "Env_list" : env_list,
              "ID": ID}

    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    from src.envs.hexapod_trossen_terrain_all.hexapod_trossen_terrain_all import Hexapod as env
    env = env(env_list, max_n_envs=3, specific_env_len=25, s_len=150)

    print(params, env.__class__.__name__)

    # Test
    if params["train"]:
        print("Training")
        policy = policies.RNN_V3_LN_PG(env, hid_dim=48, memory_dim=36, n_temp=2, tanh=params["tanh"], to_gpu=False)
        print("Model parameters: {}".format(sum(p.numel() for p in policy.parameters() if p.requires_grad)))
        #policy = policies.RNN_PG(env, hid_dim=24, tanh=params["tanh"])
        train(env, policy, params)
    else:
        policy_path = 'agents/{}_RNN_PG_R2K_pg.p'.format(env.__class__.__name__)
        policy = T.load(policy_path)
        env.test_recurrent(policy)
        print(policy_path)


