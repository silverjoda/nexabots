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

    policy_optim = T.optim.Adam(policy.parameters(), lr=params["policy_lr"], weight_decay=params["weight_decay"], eps=1e-4)

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

            # Scale rewards
            batch_rewards = (batch_rewards - batch_rewards.mean()) / batch_rewards.std()

            # Calculate episode advantages
            batch_advantages = calc_advantages_MC(params["gamma"], batch_rewards, batch_terminals)

            if params["ppo"]:
                update_ppo(policy, policy_optim, batch_states, batch_actions, batch_advantages, params["ppo_update_iters"])
            else:
                update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages)

            print("Episode {}/{}, loss_V: {}, loss_policy: {}, mean ep_rew: {}".
                  format(i, params["iters"], None, None, batch_rew / params["batchsize"])) # T.exp(policy.log_std)[0][0].detach().numpy())

            policy.decay_std(params["std_decay"])

            # Finally reset all batch lists
            batch_ctr = 0
            batch_rew = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_new_states = []
            batch_terminals = []

        if i % 500 == 0 and i > 0:
            sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "agents/{}_{}_{}_pg.p".format(env.__class__.__name__, policy.__class__.__name__, params["ID"]))
            T.save(policy.state_dict(), sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, params))

        # if i % 500 == 0 and i > 0:
        #     print("Wrote score to file")
        #     c_score, v_score, d_score = env.test(policy, N=20, seed=1337, render=False)
        #     with open("eval/{}_RE.txt".format(params["ID"]), "a+") as f:
        #         f.write("{}, {}, {}, {} \n".format(batch_rew / params["batchsize"], c_score, v_score, d_score))


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
        policy.soft_clip_grads(3.)
        policy_optim.step()

        if True:
            # Symmetry loss
            batch_states_rev = batch_states.clone()

            # Joint angles
            batch_states_rev[:, 0:3] = batch_states[:, 6:9]
            batch_states_rev[:, 3:6] = batch_states[:, 9:12]
            batch_states_rev[:, 15:18] = batch_states[:, 12:15]

            batch_states_rev[:, 6:9] = batch_states[:, 0:3]
            batch_states_rev[:, 9:12] = batch_states[:, 3:6]
            batch_states_rev[:, 12:15] = batch_states[:, 15:18]
            batch_states_rev[:, 18] = -batch_states[:, 18]

            if batch_states.shape[1] > 19:
                batch_states_rev[:, 18:24:2] = batch_states[:, 19:24:2]
                batch_states_rev[:, 19:24:2] = batch_states[:, 18:24:2]

            # Actions
            actions = policy(batch_states)
            actions_rev_pred = policy(batch_states_rev)
            actions_rev = T.zeros_like(actions)

            actions_rev[:, 0:3] = actions[:, 3:6]
            actions_rev[:, 6:9] = actions[:, 9:12]
            actions_rev[:, 12:15] = actions[:, 15:18]

            actions_rev[:, 3:6] = actions[:, 0:3]
            actions_rev[:, 9:12] = actions[:, 6:9]
            actions_rev[:, 15:18] = actions[:, 12:15]

            loss = (actions_rev_pred - actions_rev).pow(2).mean()
            print(loss)
            policy_optim.zero_grad()
            loss.backward()
            #policy.soft_clip_grads(1.)
            policy_optim.step()


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
    T.set_num_threads(2)

    env_list = ["flat"] #  ["flat", "tiles", "triangles", "holes", "pipe", "stairs", "stairs_down", "perlin"]

    if len(sys.argv) > 1:
        env_list = [sys.argv[1]]

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 500000, "batchsize": 60, "gamma": 0.99, "policy_lr": 0.0007, "weight_decay" : 0.0001, "ppo": True,
              "ppo_update_iters": 6, "animate": True, "train" : False, "env_list" : env_list,
              "note" : "Straight line with yaw", "ID" : ID, "std_decay" : 0.000, "target_vel" : 0.1, "use_contacts" : True, "turn_dir" : None}

    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    from src.envs.hexapod_trossen_terrain_all.hexapod_deploy_default import Hexapod as env
    env = env(env_list, max_n_envs=1, specific_env_len=70, s_len=120, walls=True,
              target_vel=params["target_vel"], use_contacts=params["use_contacts"], turn_dir="LEFT")

    # TODO: Experiment with RL algo improvement, add VF to PG
    # TODO: Experiment with decayed exploration
    # TODO: Try different RL algos (baselines for example)
    # TODO: Try tiles with contacts and without  (also slow vs fast speed)
    # TODO: Try training with quantized torque on legs
    # TODO: Debug yaw and contacts on real hexapod
    # TODO: Continue wiring up drone
    # TODO: Vary max torque to make policy use feedback

    # Test
    if params["train"]:
        print("Training")
        policy = policies.NN_PG(env, 96)
        print(params, env.obs_dim, env.act_dim, env.__class__.__name__, policy.__class__.__name__)
        train(env, policy, params)
    else:
        print("Testing")
        policy_name = "LH3" # Try LH3 on real robot (spoof IMU and with contacts)
        policy_path = 'agents/{}_NN_PG_{}_pg.p'.format(env.__class__.__name__, policy_name)
        policy = policies.NN_PG(env, 96)
        policy.load_state_dict(T.load(policy_path))

        env.test(policy, N=10)
        print(policy_path)





