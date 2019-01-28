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

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hid = n_hid

        self.l1 = nn.Linear(self.input_dim, self.n_hid)
        self.l2 = nn.Linear(self.n_hid, self.n_hid)
        self.l3 = nn.Linear(self.n_hid, self.output_dim)


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
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

        s_diffs = []
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
                s_diffs.append(deepcopy(s_new - s))
                s = s_new

        # Convert to torch tensors
        states_tens = torch.from_numpy(np.asarray(s_diffs, dtype=np.float32))
        state_pred_tens = torch.stack(s_diff_preds)

        # Calculate loss
        loss_states = MSE(state_pred_tens, states_tens)
        optim_state_model.zero_grad()

        # Backprop
        loss_states.backward()
        total_loss += loss_states

        # Update

        optim_state_model.step()

        if i % 1 == 0:
            print("Iter: {}/{}, states_loss: {}".format(i, iters, total_loss / BATCHSIZE))

    print("Finished pretraining model on random actions, saving")
    torch.save(state_model, '{}_state_model.pt'.format(env.__class__.__name__))



def train_opt(state_model, policy, env, iters, animate=True, lr_model=1e-3, lr_policy=2e-4, model_rpts=1):
    optim_policy = torch.optim.Adam(policy.parameters(), lr=lr_policy, weight_decay=1e-4)

    # Training algorithm:
    for i in range(iters):
        optim_policy.zero_grad()
        BATCHSIZE = 12

        total_actual_scores = 0

        score_list = []
        for j in range(BATCHSIZE):
            done = False
            s, _ = env.reset()

            sdiff_pred = T.zeros(env.obs_dim)
            curr_state = to_tensor(s, True) - sdiff_pred

            while not done:

                # Predict action from current state
                pred_a = policy(curr_state)

                # Make prediction
                sdiff_pred = state_model(T.cat([curr_state, pred_a + T.randn(policy.act_dim) * 0.1], 1))

                s, rew, done, od = env.step(pred_a.detach().numpy())
                total_actual_scores += s[27] - curr_state[:, 27]
                score_list.append(sdiff_pred[:, 27])

                # print("================")
                # print("prediction", pred_s.detach().numpy())
                # print("True", s)
                # print("Diff", np.abs(pred_s.detach().numpy() - s))
                # print("================")

                if animate:
                    env.render()

                curr_state = to_tensor(s, True) - sdiff_pred

        rp = T.cat(score_list)

        # Calculate loss
        policy_score = rp.sum()

        # Backprop
        (policy_score).backward()

        # Update
        optim_policy.step()


        print("Iter: {}/{},  predicted score: {}, actual score: {}".format(i, iters, policy_score / BATCHSIZE,
                                                                                     total_actual_scores / BATCHSIZE))

def main():
    T.set_num_threads(1)

    # Create environment
    from src.envs.centipede_mjc.centipede8_mjc_new import CentipedeMjc8 as centipede
    env = centipede()

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Create prediction model
    state_model = NN(obs_dim + act_dim, obs_dim, 64)

    # Create policy model
    policy = NN(obs_dim, act_dim, 64)

    # Pretrain model on random actions
    t1 = time.time()
    pretrain_iters = 1000
    if pretrain_iters == 0:
        state_model = torch.load("{}_state_model.pt".format(env.__class__.__name__))
        print("Loading pretrained_rnd model")
    else:
        #state_model = torch.load("{}_state_model.pt".format(env.__class__.__name__))
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

    env.test(policy)

if __name__=='__main__':
    main()