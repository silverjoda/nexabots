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
import pybullet as p

def train(env, model, params):
    model_optim = T.optim.Adam(model.parameters(), lr=params["model_lr"], weight_decay=params["weight_decay"])
    mse = T.nn.MSELoss()

    batch_states = []
    batch_actions = []
    batch_new_states = []

    batch_ctr = 0

    for i in range(params["iters"]):
        s_0 = env.reset()
        done = False
        step_ctr = 0

        latent_var = env.get_latent_label()

        while not done:
            # Sample action from policy
            action = np.random.randn(env.act_dim)

            # Step action
            s_1, _, done, _ = env.step(action.squeeze(0).numpy())
            step_ctr += 1

            if params["animate"]:
                p.removeAllUserDebugItems()
                p.addUserDebugText("sphere mass: {0:.3f}, prediction: {0:.3f}".format(env.mass, prediction), [0, 0, 2])
                env.render()

            # Record transition
            batch_states.append(my_utils.to_tensor(s_0, True))
            batch_actions.append(action)
            batch_new_states.append(my_utils.to_tensor(s_1, True))

            s_0 = s_1

        # Just completed an episode
        batch_ctr += 1

        # If enough data gathered, then perform update
        if batch_ctr == params["batchsize"]:

            batch_states = T.cat(batch_states)
            batch_actions = T.cat(batch_actions)
            batch_new_states = T.cat(batch_new_states)

            next_states_pred = model(batch_states, batch_actions)
            prediction_loss = mse(next_states_pred, batch_new_states)

            model_optim.zero_grad()
            prediction_loss.backward()
            model_optim.step()

            print("Episode {}/{}, prediction_loss: {}".
                  format(i, params["iters"], prediction_loss / params["batchsize"], ))

            # Finally reset all batch lists
            batch_ctr = 0

            batch_states = []
            batch_actions = []
            batch_new_states = []

        if i % 300 == 0 and i > 0:
            sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "agents/{}_{}_{}_pg.p".format(env.__class__.__name__, policy.__class__.__name__, params["ID"]))
            T.save(policy, sdir)
            print("Saved checkpoint at {} with params {}".format(sdir, params))

if __name__=="__main__":
    T.set_num_threads(1)

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 500000, "batchsize": 20, "gamma": 0.99, "model_lr": 0.0007, "weight_decay" : 0.0001,
              "animate": True, "train" : False,
              "note" : "Supervised model learning", "ID" : ID}

    if socket.gethostname() == "goedel":
        params["animate"] = False
        params["train"] = True

    from src.envs.cartpole_pbt.hangpole import HangPoleBulletEnv
    env = HangPoleBulletEnv(animate=params["animate"], latent_input=True, action_input=False)

    # Test
    if params["train"]:
        print("Training")
        model = policies.NN_PG(env, 16, obs_dim=env.obs_dim, act_dim=env.obs_dim)
        print(params, env.obs_dim, env.act_dim, env.__class__.__name__, model.__class__.__name__)
        train(env, model, params)
    else:
        print("Testing")
        policy_path = 'agents/HangPoleBulletEnv_NN_PG_ETX_pg.p'
        policy = T.load(policy_path)
        env.test(policy, slow=params["animate"], seed=1338)
        print(policy_path)