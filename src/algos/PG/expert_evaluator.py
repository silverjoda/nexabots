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

			# Finally reset all batch lists
			batch_ctr = 0
			batch_rew = 0

			batch_states = []
			batch_actions = []
			batch_rewards = []
			batch_new_states = []
			batch_terminals = []

		if i % 300 == 0 and i > 0:
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
		policy.soft_clip_grads(3.)
		policy_optim.step()

	if False:
		# Symmetry loss
		batch_states_rev = batch_states.clone()

		# Joint angles
		batch_states_rev[:, 0:3] = batch_states[:, 6:9]
		batch_states_rev[:, 3:6] = batch_states[:, 9:12]
		batch_states_rev[:, 15:18] = batch_states[:, 12:15]

		batch_states_rev[:, 6:9] = batch_states[:, 0:3]
		batch_states_rev[:, 9:12] = batch_states[:, 3:6]
		batch_states_rev[:, 12:15] = batch_states[:, 15:18]

		# Joint angle velocities
		batch_states_rev[:, 0 + 18:3 + 18] = batch_states[:, 6 + 18:9 + 18]
		batch_states_rev[:, 3 + 18:6 + 18] = batch_states[:, 9 + 18:12 + 18]
		batch_states_rev[:, 15 + 18:18 + 18] = batch_states[:, 12 + 18:15 + 18]

		batch_states_rev[:, 6 + 18:9 + 18] = batch_states[:, 0 + 18:3 + 18]
		batch_states_rev[:, 9 + 18:12 + 18] = batch_states[:, 3 + 18:6 + 18]
		batch_states_rev[:, 12 + 18:15 + 18] = batch_states[:, 15 + 18:18 + 18]

		# Reverse yaw and y
		batch_states_rev[44] = - batch_states[44]
		batch_states_rev[45] = - batch_states[45]

		# Reverse contacts
		batch_states_rev[46] = batch_states[48]
		batch_states_rev[47] = batch_states[49]
		batch_states_rev[51] = batch_states[50]

		batch_states_rev[48] = batch_states[46]
		batch_states_rev[49] = batch_states[47]
		batch_states_rev[50] = batch_states[51]

		# Actions
		for i in range(3):
			actions = policy(batch_states)
			actions_rev = T.zeros_like(actions)

			actions_rev[:, 0:3] = actions[:, 6:9]
			actions_rev[:, 3:6] = actions[:, 9:12]
			actions_rev[:, 15:18] = actions[:, 12:15]

			actions_rev[:, 6:9] = actions[:, 0:3]
			actions_rev[:, 9:12] = actions[:, 3:6]
			actions_rev[:, 12:15] = actions[:, 15:18]

			loss = (actions - actions_rev).pow(2).mean()
			policy_optim.zero_grad()
			loss.backward()
			policy.soft_clip_grads(1.)
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
	T.set_num_threads(1)

	env_list = ["perlin"] # ["flat", "tiles", "holes", "pipe", "inverseholes"]
	if len(sys.argv) > 1:
		env_list = [sys.argv[1]]

	ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
	params = {"iters": 500000, "batchsize": 60, "gamma": 0.995, "policy_lr": 0.0007, "weight_decay" : 0.0001, "ppo": True,
			  "ppo_update_iters": 6, "animate": True, "train" : False, "env_list" : env_list,
			  "note" : "Expert training with additional yaw and y pen", "ID" : ID}

	if socket.gethostname() == "goedel":
		params["animate"] = False
		params["train"] = True

	# TODO: Train experts (WITH ROBUST ORIENTATION RECOVERY!!!!!!!, try the yaw deviation rew)
	# TODO: Do expert comparisons
	# TODO: (THIS NOW) Find and check rnn training pipeline
	# TODO: (THIS NOW) Test torque penalization with lower pen coeff on non-flat env
	# TODO: (THIS NOW) Make expert comparison script
	# TODO: Monitor NN weights and grads during RL training

	#from src.envs.centipede.centipede import Centipede
	#env = Centipede(4)

	#from src.envs.ant_feelers_mem_mjc.ant_feelers_goal_mem_mjc import AntFeelersMjc
	#env = AntFeelersMjc()

	#from src.envs.locom_benchmarks import hf_gen
	#from src.envs.locom_benchmarks.hex_locomotion.hex_blind import Hexapod as env
	#from src.envs.locom_benchmarks.quad_locomotion.quad_blind import Quad as env
	#from src.envs.locom_benchmarks.snake_locomotion.snake_blind import Snake as env

	from src.envs.hexapod_trossen_terrain_all.hexapod_trossen_terrain_all import Hexapod as env
	#from src.envs.hexapod_trossen_terrain_all.hexapod_trossen_terrain_generalization import Hexapod as env

	#from src.envs.cartpole_pbt.hangpole import HangPoleBulletEnv as env

	# Current experts:
	# Generalization: Novar: QO6, Var: OSM
	# flat: P92, DFE
	# tiles: K4F
	# triangles: LBD
	# Stairs: HOS
	# pipe: 9GV
	# perlin: P92

	print("Testing experts")

	expert_dict = {"flat" : "DFE",
				   "tiles" : "K4F",
				   "triangles" : "LBD",
				   "stairs" : "HOS",
				   "pipe" : "9GV",
				   "perlin" : "P92"}

	results_dict = {}
	results_list = []
	results_v_list = []
	results_d_list = []
	reps = 30

	for env_name in expert_dict.keys():
		env_results_list = []
		env_results_v_list = []
		env_results_d_list = []
		env_instance = env([env_name], max_n_envs=1)
		for policy_name in expert_dict.values():

			policy_path = 'agents/{}_NN_PG_{}_pg.p'.format(env_instance.__class__.__name__, policy_name)
			policy = T.load(policy_path)

			# Evaluate
			score, v_score, d_score = env_instance.test(policy, render=False, N=reps, seed=1337)
			results_dict[env_name, policy_name] = score
			env_results_list.append(score)
			env_results_v_list.append(v_score)
			env_results_d_list.append(d_score)

		results_list.append(env_results_list)
		results_v_list.append(env_results_v_list)
		results_d_list.append(env_results_d_list)

	import pandas as pd

	# Gait quality
	print("Gait quality metric")
	pd.options.display.float_format = '{:,.2f}'.format
	dfObj = pd.DataFrame(results_list,
						 index=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'],
						 columns=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'])
	print(dfObj)

	# Velocity quality
	print("velocity quality metric")
	pd.options.display.float_format = '{:,.2f}'.format
	dfObj = pd.DataFrame(results_v_list,
						 index=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'],
						 columns=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'])
	print(dfObj)

	# Velocity quality
	print("average traveled instance")
	pd.options.display.float_format = '{:,.2f}'.format
	dfObj = pd.DataFrame(results_d_list,
						 index=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'],
						 columns=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'])
	print(dfObj)


	# Normalized results
	print("Normalized results")
	normalized_results_list = [[r / erl[i] for r in erl] for i, erl in
							   enumerate(results_list)]

	normalized_results_v_list = [[r / erl[i] for r in erl] for i, erl in
							   enumerate(results_v_list)]

	normalized_results_d_list = [[r / erl[i] for r in erl] for i, erl in
							   enumerate(results_d_list)]
	# NORMALIZED

	# Gait quality
	print("Normalized Gait quality metric")
	pd.options.display.float_format = '{:,.2f}'.format
	dfObj = pd.DataFrame(normalized_results_list,
						 index=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'],
						 columns=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'])
	print(dfObj)

	# Velocity quality
	print("Normalized velocity quality metric")
	pd.options.display.float_format = '{:,.2f}'.format
	dfObj = pd.DataFrame(normalized_results_v_list,
						 index=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'],
						 columns=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'])
	print(dfObj)

	# Velocity quality
	print("Normalized average traveled instance")
	pd.options.display.float_format = '{:,.2f}'.format
	dfObj = pd.DataFrame(normalized_results_d_list,
						 index=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'],
						 columns=['flat', 'tiles', 'triangles', 'stairs', 'pipe', 'perlin'])
	print(dfObj)