import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import time
from src.my_utils import *

from copy import deepcopy

class NN(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(NN, self).__init__()

        self.n_hid = n_hid
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.l1 = nn.Linear(self.input_dim, self.n_hid)
        self.l2 = nn.Linear(self.n_hid, self.n_hid)
        self.l3 = nn.Linear(self.n_hid, self.output_dim)


    def forward(self, x):
        x = T.tanh(self.l1(x))
        x = T.tanh(self.l2(x))
        x = T.tanh(self.l3(x))
        return x


def pretrain_model(state_model, env, iters, lr=1e-3):
    if iters == 0:
        return

    # Train prediction model on random rollouts
    MSE = torch.nn.MSELoss()
    optim_state_model = torch.optim.Adam(state_model.parameters(), lr=lr, weight_decay=1e-4)

    BATCHSIZE = 16

    for i in range(iters):
        total_loss = 0

        states = []
        s_diff_preds = []

        for j in range(BATCHSIZE):
            s, _ = env.reset()
            done = False

            while not done:
                a = np.random.randn(env.act_dim)

                # Make prediction
                sa = np.concatenate([s, a]).astype(np.float32)
                s_diff_pred = state_model(to_tensor(sa, True))
                s_diff_preds.append(s_diff_pred[0])

                s_new, rew, done, od = env.step(a)
                states.append(deepcopy(s - s_new))
                s = s_new

        # Convert to torch tensors
        states_tens = torch.from_numpy(np.asarray(states, dtype=np.float32))
        state_pred_tens = torch.stack(s_diff_preds)

        # Calculate loss
        loss_states = MSE(state_pred_tens, states_tens)

        optim_state_model.zero_grad()

        # Backprop
        loss_states.backward()
        total_loss += loss_states

        # Update
        state_model.average_grads(BATCHSIZE)
        optim_state_model.step()

        if i % 1 == 0:
            print("Iter: {}/{}, states_loss: {}".format(i, iters, total_loss / BATCHSIZE))

    print("Finished pretraining model on random actions, saving")
    torch.save(state_model, '{}_state_model.pt'.format(env.__class__.__name__))



def train_opt(state_model, policy, env, iters, animate=True, lr_model=1e-3, lr_policy=2e-4, model_rpts=1):
    optim_model = torch.optim.Adam(state_model.parameters(), lr=lr_model, weight_decay=1e-4)
    optim_policy = torch.optim.Adam(policy.parameters(), lr=lr_policy, weight_decay=1e-4)

    MSE = torch.nn.MSELoss()

    # Training algorithm:
    for i in range(iters):

        optim_policy.zero_grad()
        BATCHSIZE = 12
        total_predicted_scores = 0
        total_actual_scores = 0
        for j in range(BATCHSIZE):

            ### Policy step ----------------------------------------
            done = False
            s, _ = env.reset()
            h_p = policy.reset()
            h_s = state_model.reset()

            state_predictions = []

            pred_state = to_tensor(s, True)

            while not done:

                # Predict action from current state
                pred_a, h_p = policy(pred_state, h_p)


                # Make prediction
                pred_s, h_s = state_model(torch.cat([to_tensor(s, True), pred_a + T.randn(policy.act_dim) * 0.1], 1), h_s)
                state_predictions.append(pred_s)

                s, rew, done, od = env.step(pred_a.detach().numpy())
                total_actual_scores += s[27]

                # print("================")
                # print("prediction", pred_s.detach().numpy())
                # print("True", s)
                # print("Diff", np.abs(pred_s.detach().numpy() - s))
                # print("================")

                if animate:
                    env.render()

                # Difference between predicted state and real
                sdiff = pred_s - to_tensor(s)
                pred_state = pred_s - sdiff

            rp = T.cat(state_predictions)

            # Calculate loss
            policy_score = rp[:, 27].sum()
            total_predicted_scores += policy_score


            # Backprop
            (policy_score).backward()


        # Update
        policy.average_grads(BATCHSIZE)
        optim_policy.step()

        total_loss_states = 0

        ## Model Step ----------------------------------------
        # Backprop
        #optim_model.zero_grad()
        for j in range(model_rpts):

            done = False
            s, _ = env.reset()
            h_p = policy.reset()
            h_s = state_model.reset()

            states = []
            state_predictions = []

            while not done:

                # Predict action from current state
                with torch.no_grad():
                    pred_a, h_p = policy(to_tensor(s, True), h_p)
                    pred_a += torch.randn(1, env.act_dim) * 0.3

                # Make prediction
                pred_s, h_s = state_model(torch.cat([to_tensor(s, True), pred_a], 1), h_s)
                state_predictions.append(pred_s)

                s, rew, done, info = env.step(pred_a.numpy())
                states.append(to_tensor(s, True))

                if animate:
                    env.render()

            # Convert to torch
            states_tens = torch.cat(states)
            state_pred_tens = torch.cat(state_predictions)

            # Calculate loss
            loss_states = MSE(state_pred_tens, states_tens)
            loss_states.backward()

            total_loss_states.append(loss_states)

        # Update
        #state_model.average_grads(model_rpts)
        #optim_model.step()

        print("Iter: {}/{}, states prediction loss: {}, predicted score: {}, actual score: {}".format(i, iters,
                                                                                                      total_loss_states/BATCHSIZE,
                                                                                                      total_predicted_scores / BATCHSIZE,
                                                                                                      total_actual_scores / BATCHSIZE))

def main():
    T.set_num_threads(1)

    # Create environment
    from src.envs.centipede_mjc.centipede8_mjc_new import CentipedeMjc8 as centipede
    env = centipede()

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Create prediction model
    state_model = CM_RNN(obs_dim + act_dim, obs_dim, 64)

    # Create policy model
    policy = CM_Policy(obs_dim, act_dim, 64)

    # Pretrain model on random actions
    t1 = time.time()
    pretrain_iters = 300
    if pretrain_iters == 0:
        state_model = torch.load("{}_state_model.pt".format(env.__class__.__name__))
        print("Loading pretrained_rnd model")
    else:
        state_model = torch.load("{}_state_model.pt".format(env.__class__.__name__))
        pretrain_model(state_model, env, pretrain_iters, lr=7e-4)

    print("Pretraining finished, took {} s".format(time.time() - t1))
    torch.save(state_model, '{}_state_model.pt'.format(env.__class__.__name__))

    # Train optimization
    opt_iters = 100
    train_opt(state_model, policy, env, opt_iters, animate=True, lr_model=5e-4, lr_policy=1e-4, model_rpts=0)

    print("Finished training, saving")
    torch.save(policy, '{}_policy.pt'.format(env.__class__.__name__))


    if opt_iters == 0:
        policy = torch.load("{}_policy.pt".format(env.__class__.__name__))
        print("Loading pretrained_policy")

    env.test_recurrent(policy)

if __name__=='__main__':
    main()