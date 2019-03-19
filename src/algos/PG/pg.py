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

class Valuefun(nn.Module):
    def __init__(self, env):
        super(Valuefun, self).__init__()

        self.obs_dim = env.obs_dim

        self.fc1 = nn.Linear(self.obs_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(env, policy, params):

    policy_optim = T.optim.Adam(policy.parameters(), lr=params["policy_lr"], weight_decay=params["weight_decay"])

    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_new_states = []
    batch_terminals = []

    batch_ctr = 0
    batch_rew = 0

    for i in range(params["iters"]):
        s_0 = env.reset()
        done = False

        step_ctr = 0

        while not done:
            # Sample action from policy
            action = policy.sample_action(my_utils.to_tensor(s_0, True)).detach()

            # Step action
            s_1, r, done, _ = env.step(action.squeeze(0).numpy())
            assert r < 10, print("Large rew {}, step: {}".format(r, step_ctr))
            r = np.clip(r, -3, 3)
            step_ctr += 1

            batch_rew += r

            if params["animate"]:
                env.render()

            # Record transition
            batch_states.append(my_utils.to_tensor(s_0, True))
            batch_actions.append(action)
            batch_rewards.append(my_utils.to_tensor(np.asarray(r, dtype=np.float32), True))
            batch_new_states.append(my_utils.to_tensor(s_1, True))
            batch_terminals.append(done)

            s_0 = s_1

        # Just completed an episode
        batch_ctr += 1

        # If enough data gathered, then perform update
        if batch_ctr == params["batchsize"]:

            batch_states = T.cat(batch_states)
            batch_actions = T.cat(batch_actions)
            batch_rewards = T.cat(batch_rewards)

            # Calculate episode advantages
            batch_advantages = calc_advantages_MC(params["gamma"], batch_rewards, batch_terminals)

            if params["ppo"]:
                update_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, params["ppo_update_iters"])
            else:
                update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages)

            print("Episode {}/{}, loss_V: {}, loss_policy: {}, mean ep_rew: {}, std: {}".
                  format(i, params["iters"], None, None, batch_rew / params["batchsize"], 1)) # T.exp(policy.log_std).detach().numpy())

            # Finally reset all batch lists
            batch_ctr = 0
            batch_rew = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_new_states = []
            batch_terminals = []

        if i % 1000 == 0 and i > 0:
            sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "agents/{}_{}_{}_pg.p".format(env.__class__.__name__, policy.__class__.__name__, params["ID"]))
            T.save(policy, sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, params))


def update_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, update_iters):
    log_probs_old = policy.log_probs(batch_states, batch_actions).detach()
    c_eps = 0.2

    # Do ppo_update
    for k in range(update_iters):
        log_probs_new = policy.log_probs(batch_states, batch_actions)
        r = T.exp(log_probs_new - log_probs_old)
        loss = -T.mean(T.min(r * batch_advantages, r.clamp(1 - c_eps, 1 + c_eps) * batch_advantages))
        policy_optim.zero_grad()
        loss.backward()
        policy.soft_clip_grads(1.)
        policy_optim.step()

    #policy.print_info()


def update_V(V, V_optim, gamma, batch_states, batch_rewards, batch_terminals):
    assert len(batch_states) == len(batch_rewards) == len(batch_terminals)
    N = len(batch_states)

    # Predicted values
    Vs = V(batch_states)

    # Monte carlo estimate of targets
    targets = []
    for i in range(N):
        cumrew = T.tensor(0.)
        for j in range(i, N):
            cumrew += (gamma ** (j-i)) * batch_rewards[j]
            if batch_terminals[j]:
                break
        targets.append(cumrew.view(1, 1))

    targets = T.cat(targets)

    # MSE loss#
    V_optim.zero_grad()

    loss = (targets - Vs).pow(2).mean()
    loss.backward()
    V_optim.step()

    return loss.data


def update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages):

    # Get action log probabilities
    log_probs = policy.log_probs(batch_states, batch_actions)

    # Calculate loss function
    loss = -T.mean(log_probs * batch_advantages)

    # Backward pass on policy
    policy_optim.zero_grad()
    loss.backward()

    # Step policy update
    policy_optim.step()

    return loss.data


def calc_advantages(V, gamma, batch_states, batch_rewards, batch_next_states, batch_terminals):
    Vs = V(batch_states)
    Vs_ = V(batch_next_states)
    targets = []
    for s, r, s_, t, vs_ in zip(batch_states, batch_rewards, batch_next_states, batch_terminals, Vs_):
        if t:
            targets.append(r.unsqueeze(0))
        else:
            targets.append(r + gamma * vs_)

    return T.cat(targets) - Vs


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


    params = {"iters": 100000, "batchsize": 30, "gamma": 0.98, "policy_lr": 0.0005, "weight_decay" : 0.001, "ppo": True,
              "ppo_update_iters": 6, "animate": True, "train" : False,
              "note" : "cp, rnd, fo, 3", "ID" : ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))}
    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    #from src.envs.centipede import centipede
    #env = centipede.Centipede(4)

    #from src.envs.hexapod_flat_pd_mjc import hexapod_pd
    #env = hexapod_pd.Hexapod()

    #from src.envs.hexapod_trossen_adapt import hexapod_trossen_adapt as env
    #env = env.Hexapod()

    #from src.envs.hexapod_trossen_control import hexapod_trossen_control
    #env = hexapod_trossen_control.Hexapod()

    #from src.envs.memory_env import memory_env
    #env = memory_env.MemoryEnv()

    #from src.envs.adaptive_ctrl_env import adaptive_ctrl_env
    #env = adaptive_ctrl_env.AdaptiveSliderEnv()

    #from src.envs.hexapod_trossen_terrain import hexapod_trossen_terrain as hex_env
    #env = hex_env.Hexapod(mem_dim=0)

    from src.envs.hexapod_trossen_terrain_all import hexapod_trossen_terrain_all as hex_env
    env = hex_env.Hexapod()

    #from src.envs.hexapod_trossen_terrain_3envs import hexapod_trossen_terrain_3envs as hex_env
    #env = hex_env.Hexapod(mem_dim=0)

    #from src.envs.cartpole_swingup import cartpole_swingup
    #env = cartpole_swingup.Cartpole()

    #from src.envs.double_pendulum_swingup import double_pendulum_swinfgup
    #env = double_pendulum_swingup.Pendulum()

    # Test
    if params["train"]:
        print("Training")
        policy = policies.NN_PG(env, 16, tanh=True, scale=3)
        print(params, env.obs_dim, env.act_dim, env.__class__.__name__, policy.__class__.__name__)
        train(env, policy, params)
    else:
        print("Testing")

        p_flat = T.load('agents/Hexapod_NN_PG_TVA_pg.p')
        p_pipe = T.load('agents/Hexapod_NN_PG_WCM_pg.p')
        #p_tiles= T.load('agents/Hexapod_NN_PG_5IX_pg.p')
        #policy = T.load('agents/Hexapod_NN_PG_WCM_pg.p')

        env.test_adapt(p_flat, p_pipe, "B")