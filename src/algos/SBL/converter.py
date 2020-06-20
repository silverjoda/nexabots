import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import SubprocVecEnv
import time
import tensorflow as tf
import sys
import random
import string
import socket
import os
import torch.nn as nn
import torch as T

class PyTorchMlp(nn.Module):

  def __init__(self, n_inputs=28, n_actions=18):
      nn.Module.__init__(self)

      self.fc1 = nn.Linear(n_inputs, 64)
      self.fc2 = nn.Linear(64, 64)
      self.fc3 = nn.Linear(64, n_actions)
      self.activ_fn = nn.Tanh()
      self.out_activ = nn.Softmax(dim=0)

  def forward(self, x):
      x = self.activ_fn(self.fc1(x))
      x = self.activ_fn(self.fc2(x))
      x = self.fc3(x)
      return x



def copy_mlp_weights(baselines_model):
    torch_mlp = PyTorchMlp(n_inputs=28, n_actions=18)
    model_params = baselines_model.get_parameters()

    policy_keys = [key for key in model_params.keys() if "pi" in key]
    policy_params = [model_params[key] for key in policy_keys]

    for (th_key, pytorch_param), key, policy_param in zip(torch_mlp.named_parameters(), policy_keys, policy_params):
        param = T.from_numpy(policy_param)
        # Copies parameters from baselines model to pytorch model
        print(th_key, key)
        print(pytorch_param.shape, param.shape, policy_param.shape)
        pytorch_param.data.copy_(param.data.clone().t())

    return torch_mlp

policy_name = "H02"
policy_path = 'agents/SBL_{}'.format(policy_name)
model = A2C.load(policy_path)
print("Loading policy from: {}".format(policy_path))

for key, value in model.get_parameters().items():
  print(key, value.shape)

t_model = copy_mlp_weights(model)
sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../PG/agents/Hexapod_NN_PG_{}_pg.p'.format(policy_name))
T.save(t_model.state_dict(), sdir)
