import math
import random

import numpy as np

import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import string
import os

from copy import deepcopy

device   = torch.device("cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0


    def push(self, states, actions, rewards, next_states):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (states, actions, rewards, next_states)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        # TODO: Test this
        batch = random.sample(self.buffer, batch_size)
        statelist, actionlist, rewardlist, next_statelist = zip(*batch)

        return T.FloatTensor(statelist), \
               T.FloatTensor(actionlist), \
               T.FloatTensor(rewardlist), \
               T.FloatTensor(next_statelist)


    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.fc_in = nn.Linear(state_dim, hidden_dim)
        self.rnn = nn.LSTM(input_size=hidden_dim,
                           hidden_size=hidden_dim,
                           batch_first=True)

        self.fc_out1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out2 = nn.Linear(hidden_dim, 1)

        self.fc_out2.weight.data.uniform_(-init_w, init_w)
        self.fc_out2.bias.data.uniform_(-init_w, init_w)


    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x, _ = self.rnn(x)
        x = F.relu(self.fc_out1(x))
        x = self.fc_out2(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.fc_in = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.rnn = nn.LSTM(input_size=hidden_dim,
                           hidden_size=hidden_dim,
                           batch_first=True)

        self.fc_out1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out2 = nn.Linear(hidden_dim, 1)

        self.fc_out2.weight.data.uniform_(-init_w, init_w)
        self.fc_out2.bias.data.uniform_(-init_w, init_w)


    def forward(self, state, action):
        x = F.relu(self.fc_in(T.cat([state, action], 2)))
        x, _ = self.rnn(x)
        x = F.relu(self.fc_out1(x))
        x = self.fc_out2(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear_in = nn.Linear(num_inputs, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

        self.rnn = nn.LSTMCell(hidden_size, hidden_size)
        self.batch_rnn = nn.LSTM(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 batch_first=True)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.stale_rnn = True


    def forward(self, state):
        self.stale_rnn = True

        x = F.relu(self.linear_in(state))
        x, _ = self.batch_rnn(x)
        x = F.relu(self.linear_out(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


    def clone_params(self):
        self.rnn.bias_hh.data = deepcopy(self.batch_rnn.bias_hh_l0.data)
        self.rnn.bias_ih.data = deepcopy(self.batch_rnn.bias_ih_l0.data)
        self.rnn.weight_hh.data = deepcopy(self.batch_rnn.weight_hh_l0.data)
        self.rnn.weight_ih.data = deepcopy(self.batch_rnn.weight_ih_l0.data)


    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std


    def get_action(self, state, hidden):
        if self.stale_rnn:
            self.clone_params()
            self.stale_rnn = False

        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        x = F.relu(self.linear_in(state))
        h, c = self.rnn(x, hidden)
        x = F.relu(self.linear_out(h))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0], (h, c)


def soft_q_update(params, replay_buffer, nets, optims, criteria):
    batch_size = params["batch_size"]
    gamma = params["gamma"]
    mean_lambda = params["mean_lambda"]
    std_lambda = params["std_lambda"]
    z_lambda = params["z_lambda"]
    soft_tau = params["soft_tau"]

    value_net, target_value_net, soft_q_net, policy_net = nets
    value_optimizer, soft_q_optimizer, policy_optimizer = optims
    value_criterion, soft_q_criterion = criteria

    states, actions, rewards, next_states = replay_buffer.sample(batch_size)

    expected_q_value = soft_q_net(states, actions)
    expected_value = value_net(states)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(states)

    target_value = target_value_net(next_states)
    next_q_value = rewards.unsqueeze(2) + gamma * target_value
    q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

    expected_new_q_value = soft_q_net(states, new_action)
    next_value = expected_new_q_value - log_prob
    value_loss = value_criterion(expected_value, next_value.detach())

    log_prob_target = expected_new_q_value - expected_value
    policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

    mean_loss = mean_lambda * mean.pow(2).mean()
    std_loss = std_lambda * log_std.pow(2).mean()
    z_loss = z_lambda * z.pow(2).sum(1).mean()

    policy_loss += mean_loss + std_loss + z_loss

    soft_q_optimizer.zero_grad()
    q_value_loss.backward()
    soft_q_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )



def train(env, params):
    action_dim = env.act_dim  # env.action_space.shape[0]
    state_dim = env.obs_dim  # env.observation_space.shape[0]

    value_net = ValueNetwork(state_dim, params["hidden_dim"]).to(device)
    target_value_net = ValueNetwork(state_dim, params["hidden_dim"]).to(device)

    soft_q_net = SoftQNetwork(state_dim, action_dim, params["hidden_dim"]).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, params["hidden_dim"]).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    value_criterion = nn.MSELoss()
    soft_q_criterion = nn.MSELoss()

    value_optimizer = optim.Adam(value_net.parameters(), lr=params["value_lr"], weight_decay=0.001)
    soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=params["soft_q_lr"], weight_decay=0.001)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=params["policy_lr"], weight_decay=0.001)

    nets = (value_net, target_value_net, soft_q_net, policy_net)
    optims = (value_optimizer, soft_q_optimizer, policy_optimizer)
    criteria = (value_criterion, soft_q_criterion)

    replay_buffer = ReplayBuffer(params["replay_buffer_size"])

    rewards = []
    frame_idx = 0

    while frame_idx < params["max_frames"]:
        state = env.reset()
        episode_reward = 0
        h = None

        states = []
        actions = []
        rews = []
        next_states = []

        for step in range(params["max_steps"]):
            action, h = policy_net.get_action(state, h)
            next_state, reward, done, _ = env.step(action)

            if params["render"]:
                env.render()

            states.append(state)
            actions.append(action)
            rews.append(reward)
            next_states.append(next_state)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if frame_idx % 1000 == 0:
                print(frame_idx, rewards[-1])

            if frame_idx % 30000 == 0:
                sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "agents/{}_{}_{}_sac_r.p".format(env.__class__.__name__, policy_net.__class__.__name__,
                                                                  params["ID"]))
                T.save(policy_net, sdir)
                print("Saved checkpoint at {} with params {}".format(sdir, params))

            if done:
                break

        replay_buffer.push(states, actions, rews, next_states)
        rewards.append(episode_reward)

        if len(replay_buffer) > params["batch_size"]:
            soft_q_update(params, replay_buffer, nets, optims, criteria)

if __name__=="__main__":
    T.set_num_threads(1)

    params = {"max_frames": 8000000,
              "max_steps" : 400,
              "batch_size": 32,
              "hidden_dim": 32,
              "gamma": 0.99,
              "mean_lambda" : 1e-3,
              "std_lambda" : 1e-3,
              "z_lambda" : 0.0,
              "soft_tau" : 1e-3,
              "value_lr": 1e-4,
              "soft_q_lr": 1e-4,
              "policy_lr": 1e-5,
              "replay_buffer_size" : 1000000,
              "render": False,
              "ID" : ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))}

    # Gym env
    #import gym
    #env = gym.make("HalfCheetah-v2")

    # Centipede new
    #from src.envs.centipede_mjc.centipede8_mjc_new import CentipedeMjc8 as centipede
    #env = centipede()

    #from src.envs.hexapod_flat_mjc import hexapod
    #env = hexapod.Hexapod()

    #from src.envs.ant_feelers_mjc import ant_feelers_mjc
    #env = ant_feelers_mjc.AntFeelersMjc()

    from src.envs.hexapod_flat_pd_mjc import hexapod_pd
    env = hexapod_pd.Hexapod()

    print(params, env.__class__.__name__)

    train(env, params)

    # Testing
    #policy = T.load("agents/CentipedeMjc8_PolicyNetwork_XAS_pg.p", map_location="cpu")
    #env.test(policy)

