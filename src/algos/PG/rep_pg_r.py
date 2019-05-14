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

    policy_optim = T.optim.Adam(policy.parameters(), lr=params["lr"], weight_decay=params["decay"])
    ep_rewards = []

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
        ep_reward = 0

        # Episode lists
        episode_states = []
        episode_actions = []

        while not done:
            with T.no_grad():
                # Sample action from policy
                action, h_1 = policy.sample_action((my_utils.to_tensor(s_0, True).unsqueeze(0), h_0))

            # Step action
            s_1, r, done, _ = env.step(action[0].numpy())
            r = np.clip(r, -3, 3)

            step_ctr += 1
            episode_rew += r
            ep_reward += r

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

        ep_rewards.append(ep_reward)

        batch_states.append(T.cat(episode_states))
        batch_actions.append(T.cat(episode_actions))

        # If enough data gathered, then perform update
        if episode_ctr == params["batchsize"]:

            batch_states = T.stack(batch_states)
            batch_actions = T.stack(batch_actions)
            batch_rewards = T.cat(batch_rewards)

            # Calculate episode advantages
            batch_advantages = calc_advantages_MC(params["gamma"], batch_rewards, batch_terminals)

            if params["ppo"]:
                update_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, params["ppo_update_iters"])
            else:
                update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages)

            print("Episode {}/{}, loss_V: {}, loss_policy: {}, mean ep_rew: {}, std: {}".
                  format(i, params["iters"], None, None, episode_rew / params["batchsize"], 1)) # T.exp(policy.log_std).detach().numpy())

            # Finally reset all batch lists
            episode_ctr = 0
            episode_rew = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_terminals = []

        if i % 300 == 0 and i > 0:
            sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "agents/{}_{}_{}_pg.p".format(env.__class__.__name__, policy.__class__.__name__, params["ID"]))
            T.save(policy, sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, params))

    return ep_rewards

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
        policy.soft_clip_grads(0.5)
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
    policy.soft_clip_grads(1)
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

    env_list = [] # 177, 102, 72, -20
    if len(sys.argv) > 1:
        env_list = sys.argv[1:]
    assert len(env_list) > 0

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

    params = {"iters": 15000, "batchsize": 24, "gamma": 0.98, "lr": 0.001, "decay" : 0.0005, "ppo": True,
              "tanh" : False, "ppo_update_iters": 8, "animate": True, "train" : False,
              "comments" : "Reps, extended", "Env_list" : env_list,
              "ID": ID}

    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    #from src.envs.ant_feelers_mjc import ant_feelers_mjc
    #env = ant_feelers_mjc.AntFeelersMjc()

    #from src.envs.hexapod_flat_pd_mjc import hexapod_pd
    #env = hexapod_pd.Hexapod()

    #from src.envs.hexapod_terrain_env import hexapod_terrain
    #env = hexapod_terrain.Hexapod()

    #from src.envs.hexapod_trossen_adapt import hexapod_trossen_adapt as env
    #env = env.Hexapod()

    #from src.envs.adaptive_ctrl_env import adaptive_ctrl_env
    #env = adaptive_ctrl_env.AdaptiveSliderEnv()

    from src.envs.hexapod_trossen_terrain_all import hexapod_trossen_terrain_all as hex_env
    env = hex_env.Hexapod(env_list=env_list, max_n_envs=1)

    #from src.envs.hexapod_trossen_obstacle import hexapod_trossen_obstacle as hex_env
    #env = hex_env.Hexapod(mem_dim=0)

    #from src.envs.cartpole_swingup import cartpole_swingup
    #env = cartpole_swingup.Cartpole()

    print(params, env.__class__.__name__)

    r_lists = []
    for i in range(15):
        print("Training")
        policy = policies.RNN_V3_LN_PG(env, hid_dim=64, memory_dim=32, n_temp=3, tanh=params["tanh"], to_gpu=False)
        print("Model parameters: {}".format(sum(p.numel() for p in policy.parameters() if p.requires_grad)))
        #policy = policies.RNN_PG(env, hid_dim=24, tanh=params["tanh"])
        r_list = train(env, policy, params)
        r_lists.append(np.array(r_list))

    r_lists = np.stack(r_lists, 0)
    print(r_lists.shape)
    np.save("RNN_{}_{}".format(env_list, params["ID"]), r_lists)
