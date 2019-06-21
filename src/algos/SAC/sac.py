import math
import random

import numpy as np

import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import socket

import string
import os

device   = torch.device("cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)




class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]


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

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    expected_q_value = soft_q_net(state, action)
    expected_value = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

    target_value = target_value_net(next_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

    expected_new_q_value = soft_q_net(state, new_action)
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
    action_dim = env.act_dim#env.action_space.shape[0]
    state_dim = env.obs_dim#env.observation_space.shape[0]

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

        for step in range(params["max_steps"]):
            action = policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action)

            if params["render"]:
                env.render()

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > params["batch_size"]:
                soft_q_update(params, replay_buffer, nets, optims, criteria)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if frame_idx % 1000 == 0:
                print(frame_idx, rewards[-1])

            if frame_idx % 30000 == 0:
                sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "agents/{}_{}_{}_sac.p".format(env.__class__.__name__, policy_net.__class__.__name__,
                                                                  params["ID"]))
                T.save(policy_net, sdir)
                print("Saved checkpoint at {} with params {}".format(sdir, params))

            if done:
                break

        rewards.append(episode_reward)

if __name__=="__main__":
    T.set_num_threads(1)

    params = {"max_frames": 8000000,
              "max_steps" : 400,
              "batch_size": 64,
              "hidden_dim": 16,
              "gamma": 0.99,
              "mean_lambda" : 1e-4,
              "std_lambda" : 1e-4,
              "z_lambda" : 0.000,
              "soft_tau" : 1e-3,
              "value_lr": 1e-4,
              "soft_q_lr": 1e-4,
              "policy_lr": 1e-4,
              "replay_buffer_size" : 1000000,
              "render": False,
              "ID" : ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))}


    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    from src.envs.cartpole_pbt.cartpole_variable import CartPoleBulletEnv
    env = CartPoleBulletEnv(animate=False, latent_input=False, action_input=False)

    # from src.envs.cartpole_pbt.cartpole_mem import CartPoleBulletEnv
    # env = CartPoleBulletEnv(animate=params["animate"], latent_input=False, action_input=False)

    # from src.envs.cartpole_pbt.hangpole import HangPoleBulletEnv
    # env = HangPoleBulletEnv(animate=params["animate"], latent_input=False, action_input=False)

    # Test
    print("Training")
    train(env, params)

