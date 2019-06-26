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


def train(env, policy, latent_predictor, params):

    predictor_optim = T.optim.Adam(predictor.parameters(), lr=params["predictor_lr"], weight_decay=params["weight_decay"])

    batch_states = []
    batch_actions = []
    batch_latents = []

    batch_ctr = 0
    batch_rew = 0

    for i in range(params["iters"]):
        s_0 = env.reset()
        latent_label = env.get_latent_label()
        batch_latents.append(T.Tensor(latent_label).repeat(env.max_steps, 1))

        h_0 = None
        l_0 = np.zeros(env.latent_dim)
        done = False

        while not done:
            with T.no_grad():
                # Sample action from policy
                action = policy(my_utils.to_tensor(np.concatenate((s_0[:-1], l_0, s_0[-1:])), True))
                l_1, h_1 = latent_predictor((T.cat((my_utils.to_tensor(s_0, True), action), 1).unsqueeze(0), h_0))

            l_1 = l_1[0][0]

            # Step action
            s_1, r, done, _ = env.step(action.squeeze(0).numpy())

            # Record transition
            batch_states.append(my_utils.to_tensor(np.concatenate((s_0, l_0)), True))
            batch_actions.append(action)

            s_0 = s_1
            h_0 = h_1
            l_0 = l_1

            batch_rew += r

        # Just completed an episode
        batch_ctr += 1


        # If enough data gathered, then perform update
        if batch_ctr == params["batchsize"]:
            batch_states = T.cat(batch_states)
            batch_actions = T.cat(batch_actions)
            batch_latents = T.stack(batch_latents).float()

            batch_state_actions = T.cat((batch_states[:, :-1].view(params["batchsize"], env.max_steps, env.obs_dim),
                                         batch_actions.view(params["batchsize"], env.max_steps, env.act_dim)), 2)

            # Update latent variable model
            latent_predictions, _ = latent_predictor((batch_state_actions, None))
            latent_prediction_loss = (latent_predictions - batch_latents).pow(2).mean()
            predictor_optim.zero_grad()
            latent_prediction_loss.backward()
            predictor_optim.step()

            print("Episode {}/{}, loss_latent_pred: {:.3f}, mean ep_rew: {:.3f}, std: {:.2f}".
                  format(i, params["iters"], latent_prediction_loss, batch_rew / params["batchsize"],
                         T.exp(policy.log_std)[0][0].detach().numpy()))

            # Finally reset all batch lists
            batch_ctr = 0
            batch_rew = 0

            batch_states = []
            batch_actions = []
            batch_latents = []

        if i % 300 == 0 and i > 0:
            sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "agents/{}_{}_pg.p".format(env.__class__.__name__, params["ID"]))
            T.save(predictor, sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, params))


if __name__=="__main__":
    T.set_num_threads(1)

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 30000, "batchsize": 20, "predictor_lr": 0.0005, "weight_decay" : 0.0001,
              "train" : True, "note" : "HP, latent_estimation", "ID" : ID}

    from src.envs.cartpole_pbt.hangpole import HangPoleBulletEnv
    env = HangPoleBulletEnv(animate=False, latent_input=False, action_input=False)

    # Load ready policy
    policy = T.load('agents/HangPoleBulletEnv_NN_PG_D9T_pg.p')

    # Make predictor
    predictor = policies.RNN_PG(env, hid_dim=8, memory_dim=8, n_temp=2, obs_dim=env.obs_dim + env.act_dim,
                                       act_dim=env.latent_dim)

    # Train predictor
    train(env, policy, predictor, params)
    env.test(policy)