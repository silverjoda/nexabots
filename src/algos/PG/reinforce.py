# TAKEN FROM https://www.datahubbs.com/reinforce-with-pytorch/

import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn
from torch import optim

print(sys.version)
print(torch.__version__)
print(torch.version.cuda)




class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()

        s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]

        self.net = nn.Sequential(nn.Linear(s_dim, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, a_dim))

        # this is Variable of nn.Module, added to class automatically
        # it will be optimized as well.
        self.a_log_std = nn.Parameter(torch.zeros(1, a_dim))

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_mean = self.net(torch.FloatTensor(s))

        # [1, a_dim] => [b, a_dim]
        a_log_std = self.a_log_std.expand_as(a_mean)

        return a_mean, a_log_std

    def select_action(self, s):
        """

        :param s:
        :return:
        """
        # forward to get action mean and log_std
        # [b, s_dim] => [b, a_dim]
        a_mean, a_log_std = self.forward(s)

        # randomly sample from normal distribution, whose mean and variance come from policy network.
        # [b, a_dim]
        a = torch.normal(a_mean, torch.exp(a_log_std))

        return a

    def get_log_prob(self, s, a):
        """

        :param s:
        :param a:
        :return:
        """
        # forward to get action mean and log_std
        # [b, s_dim] => [b, a_dim]
        a_mean, a_log_std = self.forward(s)

        # [b, a_dim] => [b, 1]
        log_prob = self.normal_log_density(a, a_mean, a_log_std)

        return log_prob


    def normal_log_density(self, x, mean, log_std):
        """
        x ~ N(mean, std)
        this function will return log(prob(x)) while x belongs to guassian distrition(mean, std)
        :param x:       [b, a_dim]
        :param mean:    [b, a_dim]
        :param log_std: [b, a_dim]
        :return:        [b, 1]
        """
        std = torch.exp(log_std)
        var = std.pow(2)
        log_density = - torch.pow(x - mean, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std

        return log_density.sum(1, keepdim=True)


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r


def reinforce(env, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.98, animate=False):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.Adam(policy_estimator.net.parameters(),
                           lr=0.01)

    #action_space = np.arange(env.action_space.n)
    for ep in range(num_episodes):
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        complete = False
        while complete == False:
            # Get actions and convert to numpy array
            action = policy_estimator.select_action(np.expand_dims(s_0, 0)).detach().numpy()[0]
            s_1, r, complete, _ = env.step(action)
            if animate:
                env.render()

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # If complete, batch data
            if complete:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    # Actions are used as indices, must be LongTensor
                    action_tensor = torch.FloatTensor(batch_actions)

                    # Calculate loss

                    logprob = policy_estimator.get_log_prob(state_tensor, action_tensor)
                    loss = -(reward_tensor * logprob).mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                # Print running average
                print("\rEp: {} Average of last 10: {:.2f}".format(
                    ep + 1, np.mean(total_rewards[-10:])), end="")

    return total_rewards

env = gym.make('Hopper-v2')
s = env.reset()
pe = Policy(env)

rewards = reinforce(env, pe, num_episodes=30000, batch_size=24, animate=False)
window = 10
smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window
                    else np.mean(rewards[:i+1]) for i in range(len(rewards))]

plt.figure(figsize=(12,8))
plt.plot(rewards)
plt.plot(smoothed_rewards)
plt.ylabel('Total Rewards')
plt.xlabel('Episodes')
plt.show()