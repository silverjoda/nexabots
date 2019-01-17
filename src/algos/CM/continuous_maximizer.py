import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gym
import time
from src.policies import *
from src.my_utils import *


def pretrain_model(state_model, rew_model, env, iters, lr=1e-3):
    if iters == 0:
        return

    # Train prediction model on random rollouts
    MSE = torch.nn.MSELoss()
    optim_state_model = torch.optim.Adam(state_model.parameters(), lr=lr, weight_decay=1e-4)
    optim_rew_model = torch.optim.Adam(rew_model.parameters(), lr=lr, weight_decay=1e-4)

    for i in range(iters):
        s, _ = env.reset()
        h_s = state_model.reset()
        h_r = rew_model.reset()
        done = False

        states = []
        rewards = []
        state_predictions = []
        reward_predictions = []

        while not done:
            a = np.random.randn(env.act_dim)

            # Make prediction
            sa = np.concatenate([s, a]).astype(np.float32)
            pred_state, h_s = state_model(to_tensor(sa, True), h_s)
            pred_rew, h_r = rew_model(to_tensor(sa, True), h_r)
            state_predictions.append(pred_state[0])
            reward_predictions.append(pred_rew[0])

            s, rew, done, info = env.step(a)
            rewards.append(rew)
            states.append(s)

        # Convert to torch tensors
        states_tens = torch.from_numpy(np.asarray(states, dtype=np.float32))
        rewards_tens = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).unsqueeze(1)
        state_pred_tens = torch.stack(state_predictions)
        rew_pred_tens = torch.stack(reward_predictions)

        # Calculate loss
        loss_states = MSE(state_pred_tens, states_tens)
        loss_rewards = MSE(rew_pred_tens, rewards_tens)

        # Backprop
        optim_state_model.zero_grad()
        loss_states.backward()

        optim_rew_model.zero_grad()
        loss_rewards.backward()

        # Update
        optim_state_model.step()
        optim_rew_model.step()

        if i % 10 == 0:
            print("Iter: {}/{}, states_loss: {}, rewards_loss: {}".format(i, iters, loss_states, loss_rewards))

    print("Finished pretraining model on random actions, saving")
    torch.save(state_model, '{}_state_model.pt'.format(env.__class__.__name__))
    torch.save(rew_model, '{}_rew_model.pt'.format(env.__class__.__name__))


def train_opt(state_model, rew_model, policy, env, iters, animate=True, lr_model=1e-3, lr_policy=2e-4, model_rpts=1):
    optim_model = torch.optim.Adam(state_model.parameters(), lr=lr_model, weight_decay=1e-4)
    optim_rew = torch.optim.Adam(rew_model.parameters(), lr=lr_model, weight_decay=1e-4)
    optim_policy = torch.optim.Adam(policy.parameters(), lr=lr_policy, weight_decay=1e-4)

    MSE = torch.nn.MSELoss()

    # Training algorithm:
    for i in range(iters):

        ### Policy step ----------------------------------------
        done = False
        s, _ = env.reset()
        h_p = policy.reset()
        h_s = state_model.reset()
        h_r = rew_model.reset()

        reward_predictions = []

        sdiff = torch.zeros(1, env.obs_dim)
        pred_state = to_tensor(s, True)

        while not done:

            # Predict action from current state
            pred_a, h_p = policy(pred_state, h_p)

            # Make prediction
            pred_s, h_s = state_model(torch.cat([to_tensor(s, True), pred_a], 1), h_s)
            pred_r, h_r = state_model(torch.cat([to_tensor(s, True), pred_a], 1), h_r)
            reward_predictions.append(pred_r)

            s, rew, done, info = env.step(pred_a.detach().numpy())

            if animate:
                env.render()

            # Difference between predicted state and real
            sdiff = pred_s - to_tensor(s)

        # Convert to torch
        rew_pred_tens = torch.stack(reward_predictions)

        # Calculate loss
        policy_score = rew_pred_tens.sum()

        # Backprop
        optim_policy.zero_grad()
        (-policy_score).backward()

        # Update
        optim_policy.step()

        loss_states = 0
        loss_rewards = 0

        ## Model Step ----------------------------------------
        for j in range(model_rpts):

            done = False
            s, _ = env.reset()
            h_p = policy.reset()
            h_s = state_model.reset()
            h_r = rew_model.reset()

            states = []
            rewards = []
            state_predictions = []
            reward_predictions = []

            while not done:

                # Predict action from current state
                with torch.no_grad():
                    pred_a, h_p = policy(to_tensor(s, True), h_p)
                    pred_a += torch.randn(1, env.act_dim) * 0.3

                # Make prediction
                pred_s, h_s = state_model(torch.cat([to_tensor(s, True), pred_a], 1), h_s)
                pred_rew, h_r = rew_model(torch.cat([to_tensor(s, True), pred_a], 1), h_r)
                state_predictions.append(pred_s[0])
                reward_predictions.append(pred_rew[0])

                s, rew, done, info = env.step(pred_a.numpy())
                rewards.append(rew)
                states.append(s)

                if animate:
                    env.render()

            # Convert to torch
            states_tens = torch.from_numpy(np.asarray(states, dtype=np.float32))
            rewards_tens = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).unsqueeze(1)
            state_pred_tens = torch.stack(state_predictions)
            rew_pred_tens = torch.stack(reward_predictions)

            # Calculate loss
            loss_states = MSE(state_pred_tens, states_tens)
            loss_rewards = MSE(rew_pred_tens, rewards_tens)

            # Backprop
            optim_model.zero_grad()
            optim_rew.zero_grad()

            loss_states.backward()
            loss_rewards.backward()

            # Update
            optim_model.step()
            optim_rew.step()

        print("Iter: {}/{}, states prediction loss: {}, rew prediction loss: {}, policy score: {}".format(i, iters,
                                                                                                          loss_states,
                                                                                                          loss_rewards,
                                                                                                          policy_score))

def main():

    # Create environment
    from src.envs.centipede_mjc.centipede8_mjc_new import CentipedeMjc8 as centipede
    env = centipede()

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Create prediction model
    state_model = CM_RNN(obs_dim + act_dim, obs_dim, 64)

    # Reward prediction model
    rew_model = CM_RNN(obs_dim + act_dim, 1, 64)

    # Create policy model
    policy = CM_Policy(obs_dim, act_dim, 64)

    # Pretrain model on random actions
    t1 = time.time()
    pretrain_iters = 0
    pretrain_model(state_model, rew_model, env, pretrain_iters, lr=1e-3)
    if pretrain_iters == 0:
        state_model = torch.load("{}_state_model.pt".format(env.__class__.__name__))
        rew_model = torch.load("{}_rew_model.pt".format(env.__class__.__name__))
        print("Loading pretrained_rnd model")

    print("Pretraining finished, took {} s".format(time.time() - t1))

    # Train optimization
    opt_iters = 3000
    train_opt(state_model, rew_model, policy, env, opt_iters, animate=True, lr_model=3e-4, lr_policy=1e-3, model_rpts=0)

    print("Finished training, saving")
    torch.save(policy, '{}_policy.pt'.format(env.__class__.__name__))
    torch.save(state_model, '{}_state_model.pt'.format(env.__class__.__name__))
    torch.save(rew_model, '{}_rew_model.pt'.format(env.__class__.__name__))

    if opt_iters == 0:
        policy = torch.load("{}_policy.pt".format(env.__class__.__name__))
        print("Loading pretrained_policy")

    env.test_recurrent(policy)

if __name__=='__main__':
    main()