import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
from copy import deepcopy

class Baseline(nn.Module):
    def __init__(self, N):
        super(Baseline, self).__init__()
        self.N_links = int(N / 2)
        self.fc1 = nn.Linear(93, 40)

    def forward(self, x):
        x = self.fc1(x)
        return x


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.obs_dim = 1
        self.act_dim = 2

        self.fc1 = nn.Linear(self.obs_dim, 42)
        self.fc2 = nn.Linear(42, 42)
        self.fc3 = nn.Linear(42, self.act_dim)


    def forward(self, x):
        x = T.tanh(self.fc1(x))
        x = T.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class NN_D(nn.Module):
    def __init__(self, env):
        super(NN_D, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = 12

        self.fc1 = nn.Linear(self.obs_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)


    def forward(self, x):
        x = T.tanh(self.fc1(x))
        x = T.tanh(self.fc2(x))
        x = F.softmax(self.fc3(x))
        x = T.argmax(x, 1, keepdim=True)
        return x


class RND_D(nn.Module):
    def __init__(self, env):
        super(RND_D, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.dummy = nn.Linear(1,1)


    def forward(self, _):
        x = T.randn(1, self.act_dim)
        return T.argmax(x, 1, keepdim=True)


class RND(nn.Module):
    def __init__(self, env):
        super(RND, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.dummy = nn.Linear(1,1)


    def forward(self, _):
        return T.randn(1, self.act_dim)


    def sample_action(self, _):
        return self.forward(None)


class ConvPolicy14(nn.Module):
    def __init__(self, N):
        super(ConvPolicy14, self).__init__()
        self.N_links = int(N / 2)

        # rep conv
        self.conv_1 = nn.Conv1d(12, 6, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv1d(6, 8, kernel_size=3, stride=1)
        self.conv_3 = nn.Conv1d(8, 8, kernel_size=3, stride=1)
        self.downsample = nn.AdaptiveAvgPool1d(3)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(13, 10, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(10, 10, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(10, 6, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_3 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_4 = nn.ConvTranspose1d(18, 6, kernel_size=3, stride=1, padding=1)

        self.afun = F.tanh


    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]

        # (psi, psid)
        ext_obs = T.cat((obs[:, 3:7], obsd[:, -1:]), 1)

        # Joints angles
        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((1, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(1, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((1, 6, -1))

        jcat = T.cat((jlrs, jdlrs), 1) # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(jcat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c3 = self.afun(self.conv_3(fm_c2))

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_c3, ext_obs.unsqueeze(2)),1)))
        emb_2 = self.afun(self.conv_emb_2(emb_1))

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(emb_2))
        fm_dc2 = self.afun(self.deconv_2(fm_dc1))
        fm_dc3 = self.afun(self.deconv_3(fm_dc2))
        fm_dc4 = self.deconv_4(T.cat((fm_dc3, jcat), 1))

        acts = fm_dc4.squeeze(2).view((1, -1))

        return acts[:, 2:]


class ConvPolicy8(nn.Module):
    def __init__(self):
        super(ConvPolicy8, self).__init__()
        self.N_links = int(8 / 2)

        # rep conv
        self.conv_1 = nn.Conv1d(12, 4, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(8, 8, kernel_size=3, stride=1)
        self.conv_4 = nn.Conv1d(8, 8, kernel_size=2, stride=1)

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(13, 8, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(8, 8, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(8, 4, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(4, 4, kernel_size=3, stride=1, padding=1)
        self.deconv_3 = nn.ConvTranspose1d(4, 8, kernel_size=3, stride=1, padding=1)
        self.deconv_4 = nn.ConvTranspose1d(14, 6, kernel_size=3, stride=1, padding=1)

        self.afun = T.tanh

    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]

        # (psi, psid)
        ext_obs = T.cat((obs[:,3:7], obsd[:, -1:]), 1)

        # Joints angles
        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((1, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(1, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((1, 6, -1))

        jcat = T.cat((jlrs, jdlrs), 1) # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(jcat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c3 = self.afun(self.conv_3(fm_c2))
        fm_c4 = self.afun(self.conv_4(fm_c3))

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_c4, ext_obs.unsqueeze(2)),1)))
        emb_2 = self.afun(self.conv_emb_2(emb_1))

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(emb_2))
        fm_dc2 = self.afun(self.deconv_2(fm_dc1))
        fm_dc3 = self.afun(self.deconv_3(fm_dc2))
        fm_upsampled = F.interpolate(fm_dc3, size=4)
        fm_dc4 = self.afun(self.deconv_4(T.cat((fm_upsampled, jlrs), 1)))

        acts = fm_dc4.squeeze(2).view((1, -1))

        return acts[:, 2:]


class RecPolicy(nn.Module):
    def __init__(self, N):
        super(RecPolicy, self).__init__()

        # Amount of cells that the centipede has
        self.N_links = int(N / 2)

        # Cell RNN hidden
        self.n_hidden = 8

        # RNN for upwards pass
        self.r_up = nn.RNNCell(12, self.n_hidden)

        # Global obs
        self.fc_obs_1 = nn.Linear(13, self.n_hidden)
        self.fc_obs_2 = nn.Linear(self.n_hidden, self.n_hidden)

        # RNN for backwards pass
        self.r_down = nn.RNNCell(self.n_hidden, self.n_hidden)

        # From hidden to cell actions
        self.cell_unfc1 = nn.Linear(self.n_hidden * 2, 6)

        # Last conv layer to join with local observations
        #self.unconv_act = nn.Conv1d(3, 1, 1)

        self.afun = T.tanh


    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]
        obs_cat = T.cat((obs, obsd), 1)

        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]),1)
        jdl = T.cat((T.zeros(1, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]),1)

        h = T.zeros(1, self.n_hidden)

        h_up = []
        for i in reversed(range(self.N_links)):
            h_up.append(h)
            shift = 6 * i
            j = jl[:, shift:shift + 6]
            jd = jdl[:, shift:shift + 6]
            local_c = T.cat((j, jd), 1)
            h = self.r_up(local_c, h)

        h_up.reverse()
        h = self.afun(self.fc_obs_2(self.afun(self.fc_obs_1(obs_cat))))

        acts = []
        for i in range(self.N_links):
            shift = 6 * i
            j = jl[:, shift:shift + 6]
            jd = jdl[:, shift:shift + 6]
            jcat = T.cat((j.unsqueeze(1),jd.unsqueeze(1)), 1)


            # act_h = self.cell_unfc1(T.cat((h, h_up[i]), 1))
            # act_cat = T.cat((jcat, act_h.unsqueeze(1)), 1)
            # act_final = self.unconv_act(act_cat).squeeze(1)

            act_final = self.cell_unfc1(T.cat((h, h_up[i]), 1))
            acts.append(act_final)
            h = self.r_down(h_up[i], h)

        return T.cat(acts, 1)[:, 2:]


class StatePolicy(nn.Module):
    def __init__(self, env):
        super(StatePolicy, self).__init__()
        self.N_links = env.N_links
        self.act_dim = self.N_links * 6 - 2

        # Rep conv
        self.conv_1 = nn.Conv1d(7, 7, kernel_size=3, stride=1, padding=1)

        # Obs to state
        self.comp_mat = nn.Parameter(T.randn(1, 10, 1, 3))

        # State to action
        self.act_mat = nn.Parameter(T.randn(1, 6, 1, 2))

        # States
        self.reset()

        self.afun = T.tanh

    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]

        # (psi, psid)
        ext_rs = T.cat((obs[0,3:7].view(1,1,1,4), obsd[:, 0:1].view(1,1,1,1)), 3).repeat(1,1,self.N_links,1)

        # Joints angles
        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((1, 6, self.N_links, 1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(1, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((1, 6, self.N_links, 1))

        obscat = T.cat((T.cat((jlrs, jdlrs), 3), ext_rs), 1) # Concatenate j and jd so that they are 2 parallel channels

        comp_mat_full = self.comp_mat.repeat(1,1,self.N_links,1)
        states = self.states
        for i in range(3):
            # Concatenate observations with states
            x = T.cat((obscat, states), 3)

            # Multiply elementwise through last layer to get prestate map
            x = self.afun((x * comp_mat_full).sum(3))

            # Convolve prestate map to get new states
            states = self.afun(self.conv_1(x).unsqueeze(3))

        # Turn states into actions
        acts = self.act_mat.repeat(1,1,self.N_links,1) * T.cat((states[:,:6,:,:], jdlrs), 3)
        acts = acts.sum(3).view((1, -1))

        return acts[:, 2:]

    def reset(self):
        self.states = T.randn(1, 7, self.N_links, 1)


class PhasePolicy(nn.Module):
    def __init__(self, N):
        super(PhasePolicy, self).__init__()
        self.N_links = int(N / 2)

        # Set phase states
        self.reset()

        # Increment matrix which will be added to phases every step
        self.step_increment = T.ones(1, 6, self.N_links) * 0.01

        self.conv_obs = nn.Conv1d(10, 6, kernel_size=3, stride=1, padding=1)
        self.conv_phase = nn.Conv1d(6, 6, kernel_size=3, stride=1, padding=1)

        self.afun = T.tanh


    def step_phase(self):
        self.phases = T.fmod(self.phases + self.step_increment, 2)


    def modify_phase(self, mask):
        self.phases = T.fmod(self.phases + mask, np.pi)


    def reset(self):
        self.phases = T.randn(1, 6, self.N_links) * 0.01


    def forward(self, x):
        obs = x[:, :7]

        # (psi, psid)
        ext_rs = obs[0,3:7].view(1,4,1).repeat(1,1,self.N_links)

        # Joints angles
        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((1, 6, self.N_links))

        obscat = T.cat((jlrs, ext_rs), 1) # Concatenate j and jd so that they are 2 parallel channels

        phase_fm = self.afun(self.conv_obs(obscat))
        phase_deltas = self.afun(self.conv_phase(phase_fm))

        self.modify_phase(phase_deltas)
        self.step_phase()

        # Phases directly translate into torques
        acts = self.phases.view(1,-1) - 1

        # Phases are desired angles
        #acts = (((self.phases - (np.pi / 2)) - jlrs) * 0.1).view(1,-1)


        return acts[:, 2:]


class ConvPolicy_Iter_PG(nn.Module):
    def __init__(self, env):
        super(ConvPolicy_Iter_PG, self).__init__()
        self.N_links = env.N_links
        self.act_dim = self.N_links * 6 - 2

        # rep conv
        self.conv_1 = nn.Conv1d(17, 6, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(6, 6, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(6, 6, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv1d(12, 6, kernel_size=3, stride=1, padding=1)
        #self.conv_5 = nn.Conv1d(6, 6, kernel_size=3, stride=1, padding=1)

        self.afun = F.selu
        self.log_std = T.zeros(1, self.act_dim)

    def forward(self, x):
        M = x.shape[0]
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]

        # (psi, psid)
        ext_obs = T.cat((obs[:, 3:7], obsd[:, -1:]), 1).unsqueeze(2)
        ext_obs_rep = ext_obs.repeat((1, 1, self.N_links))

        # Joints angles
        jl = T.cat((T.zeros(M, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((M, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(M, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((M, 6, -1))

        ocat = T.cat((jlrs, jdlrs, ext_obs_rep), 1)  # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(ocat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c3 = self.afun(self.conv_3(fm_c2))
        fm_c4 = self.conv_4(T.cat((fm_c3, jlrs), 1))

        acts = fm_c4.squeeze(2).view((M, -1))

        return acts[:, 2:]


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class ConvPolicy_Iter_PG_new(nn.Module):
    def __init__(self, env):
        super(ConvPolicy_Iter_PG_new, self).__init__()
        self.N_links = env.N_links
        self.act_dim = self.N_links * 6 - 2

        # rep conv
        self.conv_1 = nn.Conv1d(20, 6, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(6, 6, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(6, 6, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv1d(14, 6, kernel_size=3, stride=1, padding=1)

        self.afun = F.selu
        self.log_std = T.zeros(1, self.act_dim)

    def forward(self, x):
        # Batch dimension
        M = x.shape[0]

        # z, qw, qx, qy, qz [b,5]
        obs = x[:, :5]

        # xd, yd, xz, xangd, yangd, zangd [b, 6]
        obsd = x[:, 5 + self.N_links * 6 - 2: 5 + self.N_links * 6 - 2 + 6]

        # qw, qx, qy, qz, xd, yd [b, 6]
        ext_obs = T.cat((obs[:, 1:5], obsd[:, 0:2]), 1).unsqueeze(2)
        ext_obs_rep = ext_obs.repeat((1, 1, self.N_links))

        # Joints angles
        jl = T.cat((T.zeros(M, 2), x[:, 5:5 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((M, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(M, 2), x[:, 5 + self.N_links * 6 - 2 + 6:5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2]), 1)
        jdlrs = jdl.view((M, 6, -1))

        # Contacts
        jcl = x[:, 5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2:]
        jclrs = jcl.view((M, 2, -1))

        ocat = T.cat((jlrs, jdlrs, ext_obs_rep, jclrs), 1)  # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(ocat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c3 = self.afun(self.conv_3(fm_c2))
        fm_c4 = self.conv_4(T.cat((fm_c3, jlrs, jclrs), 1))

        acts = fm_c4.squeeze(2).view((M, -1))

        return acts[:, 2:]


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class ConvPolicy_Iter_PG_c(nn.Module):
    def __init__(self, env):
        super(ConvPolicy_Iter_PG_c, self).__init__()
        self.N_links = env.N_links
        self.act_dim = self.N_links * 6 - 2

        # rep conv
        self.conv_1 = nn.Conv1d(20, 6, kernel_size=3, stride=1, padding=1)

        self.afun = F.selu
        self.log_std = T.zeros(1, self.act_dim)

    def forward(self, x):
        # Batch dimension
        M = x.shape[0]

        # z, qw, qx, qy, qz [b,5]
        obs = x[:, :5]

        # xd, yd, xz, xangd, yangd, zangd [b, 6]
        obsd = x[:, 5 + self.N_links * 6 - 2: 5 + self.N_links * 6 - 2 + 6]

        # qw, qx, qy, qz, xd, yd [b, 6]
        ext_obs = T.cat((obs[:, 1:5], obsd[:, 0:2]), 1).unsqueeze(2)
        ext_obs_rep = ext_obs.repeat((1, 1, self.N_links))

        # Joints angles
        jl = T.cat((T.zeros(M, 2), x[:, 5:5 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((M, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(M, 2), x[:, 5 + self.N_links * 6 - 2 + 6:5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2]), 1)
        jdlrs = jdl.view((M, 6, -1))

        # Contacts
        jcl = x[:, 5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2:]
        jclrs = jcl.view((M, 2, -1))

        ocat = T.cat((jlrs, jdlrs, ext_obs_rep, jclrs), 1)  # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = T.tanh(self.conv_1(ocat))

        acts = fm_c1.squeeze(2).view((M, -1))

        return acts[:, 2:]


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class ConvPolicy8_PG(nn.Module):
    def __init__(self, env, N_neurons, tanh=False, std_fixed=True):
        super(ConvPolicy8_PG, self).__init__()
        self.N_links = 4
        self.act_dim = self.N_links * 6 - 2

        # rep conv
        self.conv_1 = nn.Conv1d(14, 12, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(12, 12, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(12, 12, kernel_size=3, stride=1)
        self.conv_4 = nn.Conv1d(12, 12, kernel_size=2, stride=1)

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(22, 12, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(12, 12, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(12, 12, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(12, 12, kernel_size=3, stride=1, padding=1)
        self.deconv_3 = nn.ConvTranspose1d(12, 12, kernel_size=3, stride=1, padding=1)
        self.deconv_4 = nn.ConvTranspose1d(18, 6, kernel_size=3, stride=1, padding=1)
        #
        # T.nn.init.kaiming_normal_(self.conv_1.weight, mode='fan_in', nonlinearity='leaky_relu')
        # T.nn.init.kaiming_normal_(self.conv_2.weight, mode='fan_in', nonlinearity='leaky_relu')
        # T.nn.init.kaiming_normal_(self.conv_3.weight, mode='fan_in', nonlinearity='leaky_relu')
        # T.nn.init.kaiming_normal_(self.conv_4.weight, mode='fan_in', nonlinearity='leaky_relu')
        #
        # T.nn.init.kaiming_normal_(self.deconv_1.weight, mode='fan_in', nonlinearity='leaky_relu')
        # T.nn.init.kaiming_normal_(self.deconv_2.weight, mode='fan_in', nonlinearity='leaky_relu')
        # T.nn.init.kaiming_normal_(self.deconv_3.weight, mode='fan_in', nonlinearity='leaky_relu')
        # T.nn.init.kaiming_normal_(self.deconv_4.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.upsample = nn.Upsample(size=4)

        self.afun = F.selu

        self.log_std = T.zeros(1, self.act_dim)

    def forward(self, x):
        N_joints = self.N_links * 6 - 2
        N = x.shape[0]


        # (psi, psid)
        global_obs = T.cat((x[:, N_joints*2 + self.N_links * 2:N_joints*2 + self.N_links * 2 + 4],
                            x[:, N_joints*2 + self.N_links * 2 + 4:]), 1)

        # Joints angles
        jl = T.cat((T.zeros(N, 2), x[:, :N_joints]), 1)
        jlrs = jl.view((N, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(N, 2), x[:, N_joints:N_joints*2]), 1)
        jdlrs = jdl.view((N, 6, -1))

        # Contacts
        contacts = x[:, N_joints * 2:N_joints * 2 + self.N_links * 2]
        jcrs = contacts.view((N, 2, -1))

        jcat = T.cat((jlrs, jdlrs, jcrs), 1)  # Concatenate j, jd and contacts so that they are 3 parallel channels

        fm_c1 = self.afun(self.conv_1(jcat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c3 = self.afun(self.conv_3(fm_c2))
        fm_c4 = self.afun(self.conv_4(fm_c3))

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_c4, global_obs.unsqueeze(2)), 1)))
        emb_2 = self.afun(self.conv_emb_2(emb_1))

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(emb_2))
        fm_dc2 = self.afun(self.deconv_2(fm_dc1))
        fm_dc3 = self.afun(self.deconv_3(fm_dc2))
        fm_upsampled = F.interpolate(fm_dc3, size=4)
        fm_dc4 = self.deconv_4(T.cat((fm_upsampled, jlrs), 1))

        acts = fm_dc4.squeeze(2).view((N, -1))

        return acts[:, 2:]


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            # print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd



    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class ConvPolicy14_PG(nn.Module):
    def __init__(self, env):
        super(ConvPolicy14_PG, self).__init__()
        self.N_links = 7

        self.act_dim = self.N_links * 6 - 2

        # rep conv
        self.conv_1 = nn.Conv1d(12, 6, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv1d(6, 8, kernel_size=3, stride=1)
        self.conv_3 = nn.Conv1d(8, 8, kernel_size=3, stride=1)
        self.downsample = nn.AdaptiveAvgPool1d(3)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(13, 10, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(10, 10, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(10, 6, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_3 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_4 = nn.ConvTranspose1d(18, 6, kernel_size=3, stride=1, padding=1)

        self.afun = F.selu

        self.log_std = T.zeros(1, self.act_dim)

    def forward(self, x):
        M = x.shape[0]
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]
        self.log_std = T.zeros(1, self.act_dim)
        # (psi, psid)
        ext_obs = T.cat((obs[:, 3:7], obsd[:, -1:]), 1)

        # Joints angles
        jl = T.cat((T.zeros(M, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((M, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(M, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((M, 6, -1))

        jcat = T.cat((jlrs, jdlrs), 1) # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(jcat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c3 = self.afun(self.conv_3(fm_c2))

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_c3, ext_obs.unsqueeze(2)),1)))
        emb_2 = self.afun(self.conv_emb_2(emb_1))

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(emb_2))
        fm_dc2 = self.afun(self.deconv_2(fm_dc1))
        fm_dc3 = self.afun(self.deconv_3(fm_dc2))
        fm_dc4 = self.deconv_4(T.cat((fm_dc3, jcat), 1))

        acts = fm_dc4.squeeze(2).view((M, -1))

        return acts[:, 2:]


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class ConvPolicy30_PG(nn.Module):
    def __init__(self, env):
        super(ConvPolicy30_PG, self).__init__()
        self.N_links = 15
        self.act_dim = self.N_links * 6 - 2

        # rep conv
        self.conv_1 = nn.Conv1d(12, 6, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv1d(6, 8, kernel_size=3, stride=1)
        self.conv_3 = nn.Conv1d(8, 8, kernel_size=3, stride=1)
        self.downsample = nn.AdaptiveAvgPool1d(5)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(13, 10, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(10, 10, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(10, 6, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_3 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_4 = nn.ConvTranspose1d(18, 6, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=13)

        self.afun = F.selu

        self.log_std = T.zeros(1, self.act_dim)


    def forward(self, x):
        M = x.shape[0]
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]

        # (psi, psid)
        ext_obs = T.cat((obs[:, 3:7], obsd[:, -1:]), 1)

        # Joints angles
        jl = T.cat((T.zeros(M, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((M, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(M, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((M, 6, -1))

        jcat = T.cat((jlrs, jdlrs), 1) # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(jcat))
        fm_c1_ds = self.downsample(fm_c1)
        fm_c2 = self.afun(self.conv_2(fm_c1_ds))
        fm_c3 = self.afun(self.conv_3(fm_c2))

        # Avg pool through link channels
        fm_links = self.pool(fm_c3) # (1, N, 1)

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_links, ext_obs.unsqueeze(2)),1)))
        emb_2 = self.afun(self.conv_emb_2(emb_1))

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(emb_2))
        fm_dc2 = self.afun(self.deconv_2(fm_dc1))
        fm_dc2_us = self.upsample(fm_dc2)
        fm_dc3 = self.afun(self.deconv_3(fm_dc2_us))
        fm_dc4 = self.deconv_4(T.cat((fm_dc3, jcat), 1)) # Change jcat to jlrs

        acts = fm_dc4.squeeze(2).view((M, -1))

        return acts[:, 2:]


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class Linear_PG(nn.Module):
    def __init__(self, env, hid_dim=64, tanh=False, std_fixed=True, obs_dim=None, act_dim=None):
        super(Linear_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        if obs_dim is not None:
            self.obs_dim = obs_dim

        if act_dim is not None:
            self.act_dim = act_dim

        self.fc1 = nn.Linear(self.obs_dim, self.act_dim)

        if std_fixed:
            self.log_std = T.zeros(1, self.act_dim)
        else:
            self.log_std = nn.Parameter(T.zeros(1, self.act_dim))


    def forward(self, x):
        x = self.fc1(x)
        return x


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            # print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class NN_PG(nn.Module):
    def __init__(self, env, hid_dim=64, tanh=False, std_fixed=True, obs_dim=None, act_dim=None):
        super(NN_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        if obs_dim is not None:
            self.obs_dim = obs_dim

        if act_dim is not None:
            self.act_dim = act_dim

        self.tanh = tanh

        self.fc1 = nn.Linear(self.obs_dim, hid_dim)
        self.m1 = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.m2 = nn.LayerNorm(hid_dim)
        self.fc3 = nn.Linear(hid_dim, self.act_dim)

        T.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='linear')

        if std_fixed:
            self.log_std = T.zeros(1, self.act_dim)
        else:
            self.log_std = nn.Parameter(T.zeros(1, self.act_dim))


    def decay_std(self, decay=0.002):
        self.log_std -= decay


    def forward(self, x):
        x = F.leaky_relu(self.m1(self.fc1(x)))
        x = F.leaky_relu(self.m2(self.fc2(x)))
        if self.tanh:
            x = T.tanh(self.fc3(x))
        else:
            x = self.fc3(x)
        return x


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m


        if maxval > bnd:
            # print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class NN_PG_SB(nn.Module):
    def __init__(self, env, hid_dim=64, tanh=False, std_fixed=True, obs_dim=None, act_dim=None):
        super(NN_PG_SB, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        if obs_dim is not None:
            self.obs_dim = obs_dim

        if act_dim is not None:
            self.act_dim = act_dim

        self.tanh = tanh

        self.fc1 = nn.Linear(self.obs_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, self.act_dim)

        T.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='linear')

        if std_fixed:
            self.log_std = T.zeros(1, self.act_dim)
        else:
            self.log_std = nn.Parameter(T.zeros(1, self.act_dim))


    def decay_std(self, decay=0.002):
        self.log_std -= decay


    def forward(self, x):
        x = T.tanh(self.fc1(x))
        x = T.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m


        if maxval > bnd:
            # print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)



class NN_PG_VF(nn.Module):
    def __init__(self, env, hid_dim=64, tanh=False, std_fixed=True, obs_dim=None, act_dim=None):
        super(NN_PG_VF, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        if obs_dim is not None:
            self.obs_dim = obs_dim

        if act_dim is not None:
            self.act_dim = act_dim

        self.tanh = tanh

        self.fc1 = nn.Linear(self.obs_dim, hid_dim)
        self.m1 = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.m2 = nn.LayerNorm(hid_dim)
        self.fc3 = nn.Linear(hid_dim, self.act_dim)

        T.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='linear')

        if std_fixed:
            self.log_std = T.zeros(1, self.act_dim)
        else:
            self.log_std = nn.Parameter(T.zeros(1, self.act_dim))


    def decay_std(self, decay=0.002):
        self.log_std -= decay


    def forward(self, x):
        x = F.leaky_relu(self.m1(self.fc1(x)))
        x = F.leaky_relu(self.m2(self.fc2(x)))
        if self.tanh:
            x = T.tanh(self.fc3(x))
        else:
            x = self.fc3(x)
        return x


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m


        if maxval > bnd:
            # print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class NN_PG_CONVMEM(nn.Module):
    def __init__(self, env, hid_dim=32, tanh=False, std_fixed=True):
        super(NN_PG_CONVMEM, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.env = env
        self.tanh = tanh

        self.conv1 = nn.Conv1d(in_channels=self.obs_dim, out_channels=6, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=4, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=0)

        self.fc1 = nn.Linear(self.obs_dim + 4 * 5, hid_dim)
        self.m1 = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, self.act_dim)

        if std_fixed:
            self.log_std = T.zeros(1, self.act_dim)
        else:
            self.log_std = nn.Parameter(T.zeros(1, self.act_dim))


    def forward(self, x):
        x_mem = x.view(-1, self.obs_dim, self.env.mem_horizon)
        x = x_mem[:, :, -1]

        # Do convolution on the memory
        x_mem = F.relu(self.conv1(x_mem))
        x_mem = F.relu(self.conv2(x_mem))
        x_mem = F.relu(self.conv3(x_mem))
        x_mem = x_mem.view(-1, 20)

        # Concat and FF
        x_c = F.relu(self.m1(self.fc1(T.cat([x_mem, x], 1))))

        if self.tanh:
            x_c = T.tanh(self.fc2(x_c))
        else:
            x_c = self.fc2(x_c)
        return x_c


    def soft_clip_grads(self, bnd=1):

        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m


        if maxval > bnd:
            # print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class NN_PG_SHORT(nn.Module):
    def __init__(self, env, hid_dim=64, tanh=False, std_fixed=True):
        super(NN_PG_SHORT, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.tanh = tanh
        #self.scale = scale

        self.fc1 = nn.Linear(self.obs_dim, hid_dim)
        self.m1 = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, self.act_dim)

        if std_fixed:
            self.log_std = T.zeros(1, self.act_dim)
        else:
            self.log_std = nn.Parameter(T.zeros(1, self.act_dim))


    def forward(self, x):
        x = F.relu(self.m1(self.fc1(x)))
        if self.tanh:
            x = T.tanh(self.fc2(x))
        else:
            x = self.fc2(x)
        return x


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            # print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class NN_PG_STD(nn.Module):
    def __init__(self, env, hid_dim=64, tanh=False):
        super(NN_PG_STD, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.tanh = tanh

        self.fc1 = nn.Linear(self.obs_dim, hid_dim)
        self.m1 = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.m2 = nn.LayerNorm(hid_dim)
        self.fc3 = nn.Linear(hid_dim, self.act_dim)


    def forward(self, x):
        x = F.relu(self.m1(self.fc1(x)))
        x = F.relu(self.m2(self.fc2(x)))
        if self.tanh:
            x = T.tanh(self.fc3(x))
        else:
            x = self.fc3(x)
        return x


    def print_info(self):
        print("-------------------------------")
        print("w_fc1", self.fc1.weight.data.max(), self.fc1.weight.data.min())
        print("b_fc1", self.fc1.bias.data.max(), self.fc1.weight.data.min())
        print("w_fc2", self.fc2.weight.data.max(), self.fc2.weight.data.min())
        print("b_fc2", self.fc2.bias.data.max(), self.fc2.weight.data.min())
        print("w_fc3", self.fc3.weight.data.max(), self.fc3.weight.data.min())
        print("b_fc3", self.fc3.bias.data.max(), self.fc3.weight.data.min())
        print("---")
        print("w_fc1 grad", self.fc1.weight.grad.max(), self.fc1.weight.grad.min())
        print("b_fc1 grad", self.fc1.bias.grad.max(), self.fc1.bias.grad.min())
        print("w_fc2 grad", self.fc2.weight.grad.max(), self.fc2.weight.grad.min())
        print("b_fc2 grad", self.fc2.bias.grad.max(), self.fc2.bias.grad.min())
        print("w_fc3 grad", self.fc3.weight.grad.max(), self.fc3.weight.grad.min())
        print("b_fc3 grad", self.fc3.bias.grad.max(), self.fc3.bias.grad.min())
        print("-------------------------------")


    def clip_grads(self, bnd=1):
        self.fc1.weight.grad.clamp_(-bnd, bnd)
        self.fc1.bias.grad.clamp_(-bnd, bnd)
        self.fc2.weight.grad.clamp_(-bnd, bnd)
        self.fc2.bias.grad.clamp_(-bnd, bnd)
        self.fc3.weight.grad.clamp_(-bnd, bnd)
        self.fc3.bias.grad.clamp_(-bnd, bnd)


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            print("Soft clipping gradients")
            self.fc1.weight.grad = (self.fc1.weight.grad / maxval) * bnd
            self.fc1.bias.grad = (self.fc2.bias.grad / maxval) * bnd
            self.fc2.weight.grad = (self.fc2.weight.grad / maxval) * bnd
            self.fc2.bias.grad = (self.fc2.bias.grad / maxval) * bnd
            self.fc3.weight.grad = (self.fc3.weight.grad / maxval) * bnd
            self.fc3.bias.grad = (self.fc3.bias.grad / maxval) * bnd


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class NN_PREC_PG(nn.Module):
    def __init__(self, env, hid_dim=64, tanh=False):
        super(NN_PREC_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.tanh = tanh

        self.fc1 = nn.Linear(self.obs_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.act_dim)
        self.fc_f = nn.Linear(self.obs_dim + self.act_dim, self.act_dim)

        self.log_std = T.zeros(1, self.act_dim)


    def forward(self, input):
        x = F.selu(self.fc1(input))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))

        if self.tanh:
            x = T.tanh(self.fc_f(T.cat((input, x), 1)))
        else:
            x = self.fc_f(T.cat((input, x), 1))
        return x


    def print_info(self):
        print("-------------------------------")
        print("w_fc1", self.fc1.weight.data.max(), self.fc1.weight.data.min())
        print("b_fc1", self.fc1.bias.data.max(), self.fc1.weight.data.min())
        print("w_fc2", self.fc2.weight.data.max(), self.fc2.weight.data.min())
        print("b_fc2", self.fc2.bias.data.max(), self.fc2.weight.data.min())
        print("w_fc3", self.fc3.weight.data.max(), self.fc3.weight.data.min())
        print("b_fc3", self.fc3.bias.data.max(), self.fc3.weight.data.min())
        print("---")
        print("w_fc1 grad", self.fc1.weight.grad.max(), self.fc1.weight.grad.min())
        print("b_fc1 grad", self.fc1.bias.grad.max(), self.fc1.bias.grad.min())
        print("w_fc2 grad", self.fc2.weight.grad.max(), self.fc2.weight.grad.min())
        print("b_fc2 grad", self.fc2.bias.grad.max(), self.fc2.bias.grad.min())
        print("w_fc3 grad", self.fc3.weight.grad.max(), self.fc3.weight.grad.min())
        print("b_fc3 grad", self.fc3.bias.grad.max(), self.fc3.bias.grad.min())
        print("-------------------------------")


    def clip_grads(self, bnd=1):
        self.fc1.weight.grad.clamp_(-bnd, bnd)
        self.fc1.bias.grad.clamp_(-bnd, bnd)
        self.fc2.weight.grad.clamp_(-bnd, bnd)
        self.fc2.bias.grad.clamp_(-bnd, bnd)
        self.fc3.weight.grad.clamp_(-bnd, bnd)
        self.fc3.bias.grad.clamp_(-bnd, bnd)


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            print("Soft clipping gradients")
            self.fc1.weight.grad = (self.fc1.weight.grad / maxval) * bnd
            self.fc1.bias.grad = (self.fc1.bias.grad / maxval) * bnd
            self.fc2.weight.grad = (self.fc2.weight.grad / maxval) * bnd
            self.fc2.bias.grad = (self.fc2.bias.grad / maxval) * bnd
            self.fc3.weight.grad = (self.fc3.weight.grad / maxval) * bnd
            self.fc3.bias.grad = (self.fc3.bias.grad / maxval) * bnd


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class NN_PG_MICRO(nn.Module):
    def __init__(self, env):
        super(NN_PG_MICRO, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.fc1 = nn.Linear(self.obs_dim, 24)
        self.fc2 = nn.Linear(24, self.act_dim)

        self.log_std = T.zeros(1, self.act_dim)


    def forward(self, x):
        x = T.tanh(self.fc1(x))
        x = T.tanh(self.fc2(x))
        return x


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class NN_PG_D(nn.Module):
    def __init__(self, env):
        super(NN_PG_D, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.hid_dim = 8
        self.fc1 = nn.Linear(self.obs_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)


    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


    def sample_action(self, s):
        x = self.forward(s)
        return x.multinomial(1)


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_softmax = self.forward(batch_states)
        return T.log(action_softmax.gather(1, batch_actions.long()))


class CNN_PG(nn.Module):
    def __init__(self, env):
        super(CNN_PG, self).__init__()
        self.obs_dim = env.obs_dim - (24**2 * 2)
        self.act_dim = env.act_dim

        self.conv1 = nn.Conv2d(2, 8, 5, 1, 2)
        self.conv2 = nn.Conv2d(8, 8, 5, 1, 2)
        self.conv3 = nn.Conv2d(8, 8, 3, 1, 1)

        self.fcc = nn.Linear(6 * 6 * 8, 32)
        self.fcj = nn.Bilinear(32, self.obs_dim, 48)

        self.fc1 = nn.Linear(48, 48)
        self.fc2 = nn.Linear(48, self.act_dim)

        #self.log_std = nn.Parameter(T.zeros(1, self.act_dim))
        self.log_std = T.zeros(1, self.act_dim)


    def forward(self, x):
        x_obs = x[:, :self.obs_dim]
        x_img = x[:, self.obs_dim:].view(x.shape[0], 2, 24, 24)

        c = F.avg_pool2d(F.selu(self.conv1(x_img)), 2)
        c = F.avg_pool2d(F.selu(self.conv2(c)), 2)
        c = F.selu(self.conv3(c))
        c = c.view(x.shape[0], 6 * 6 * 8)
        c = F.selu(self.fcc(c))

        x = self.fcj(c, x_obs)

        x = F.selu(self.fc1(x))
        x = self.fc2(x)
        return x


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class RNN_V3_PG(nn.Module):
    def __init__(self, env, hid_dim=64, memory_dim=64, n_temp=3, tanh=False, to_gpu=False):
        super(RNN_V3_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.tanh = tanh
        self.to_gpu = to_gpu

        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.fc2 = nn.Linear(self.obs_dim + self.memory_dim, self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)

        if to_gpu:
            self.log_std_gpu = T.zeros(1, self.act_dim).cuda()
        else:
            self.log_std_cpu = T.zeros(1, self.act_dim)


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            #print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input
        rnn_input = F.selu(self.fc1(x))
        output, h = self.rnn(rnn_input, h)
        f = F.selu(self.fc2(T.cat((output, x), 2)))
        if self.tanh:
            f = T.tanh(self.fc3(f))
        else:
            f = self.fc3(f)
        return f, h


    def sample_action(self, s):
        x, h = self.forward(s)
        return T.normal(x[0], T.exp(self.log_std_cpu)), h


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward((batch_states, None))

        # Calculate probabilities
        if self.to_gpu:
            log_std_batch = self.log_std_gpu.expand_as(action_means)
        else:
            log_std_batch = self.log_std_cpu.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


class RNN_V3_LN_PG(nn.Module):
    def __init__(self, env, hid_dim=64, memory_dim=64, n_temp=3, tanh=False, to_gpu=False):
        super(RNN_V3_LN_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.tanh = tanh
        self.to_gpu = to_gpu

        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.m1 = nn.LayerNorm(self.obs_dim)
        self.fc2 = nn.Linear(self.obs_dim + self.memory_dim, self.hid_dim)
        self.m2 = nn.LayerNorm(self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)

        if to_gpu:
            self.log_std_gpu = T.zeros(1, self.act_dim).cuda()
        else:
            self.log_std_cpu = T.zeros(1, self.act_dim)


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            #print("Soft clipping grads")

            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input
        rnn_features = F.selu(self.m1(self.fc1(x)))

        output, h = self.rnn(rnn_features, h)

        f = F.selu(self.m2(self.fc2(T.cat((output, x), 2))))
        if self.tanh:
            f = T.tanh(self.fc3(f))
        else:
            f = self.fc3(f)
        return f, h


    def sample_action(self, s):
        x, h = self.forward(s)
        return T.normal(x[0], T.exp(self.log_std_cpu)), h


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward((batch_states, None))

        # Calculate probabilities
        if self.to_gpu:
            log_std_batch = self.log_std_gpu.expand_as(action_means)
        else:
            log_std_batch = self.log_std_cpu.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


class RNN_PG(nn.Module):
    def __init__(self, env, hid_dim=64, memory_dim=64, n_temp=3, obs_dim=None, act_dim=None, tanh=False, to_gpu=False):
        super(RNN_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        if obs_dim is not None:
            self.obs_dim = obs_dim

        if act_dim is not None:
            self.act_dim = act_dim

        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.tanh = tanh
        self.to_gpu = to_gpu

        # In order of application
        self.fc1 = nn.Linear(self.obs_dim, self.hid_dim)
        self.m1 = nn.LayerNorm(self.hid_dim)

        self.rnn = nn.LSTM(self.hid_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc2 = nn.Linear(self.memory_dim, self.act_dim)

        T.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')

        if to_gpu:
            self.log_std_gpu = T.zeros(1, self.act_dim).cuda()
        else:
            self.log_std_cpu = T.zeros(1, self.act_dim)


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            #print("Soft clipping grads")

            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input
        rnn_features = F.leaky_relu(self.m1(self.fc1(x)))

        output, h = self.rnn(rnn_features, h)

        if self.tanh:
            f = T.tanh(self.fc3(output))
        else:
            f = self.fc2(output)
        return f, h


    def forward_hidden(self, input):
        x, h = input
        rnn_features = F.leaky_relu(self.m1(self.fc1(x)))

        N = x.shape[1]
        hiddens = [h]
        for i in range(N):
            _, h = self.rnn(rnn_features[:, i:i+1, :], h)
            hiddens.append(h)
        return hiddens[:-1]


    def sample_action(self, s):
        x, h = self.forward(s)
        return T.normal(x[0], T.exp(self.log_std_cpu)), h


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward((batch_states, None))

        # Calculate probabilities
        if self.to_gpu:
            log_std_batch = self.log_std_gpu.expand_as(action_means)
        else:
            log_std_batch = self.log_std_cpu.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


    def log_probs_wh(self, h, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward((batch_states, h))

        # Calculate probabilities
        if self.to_gpu:
            log_std_batch = self.log_std_gpu.expand_as(action_means)
        else:
            log_std_batch = self.log_std_cpu.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


class RNN_C_PG(nn.Module):
    def __init__(self, env, hid_dim=64, memory_dim=64, n_temp=3, tanh=False, to_gpu=False):
        super(RNN_C_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.tanh = tanh
        self.to_gpu = to_gpu

        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.m1 = nn.LayerNorm(self.obs_dim)
        self.fc2 = nn.Linear(self.memory_dim, self.hid_dim)
        self.m2 = nn.LayerNorm(self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)

        if to_gpu:
            self.log_std_gpu = T.zeros(1, self.act_dim).cuda()
        else:
            self.log_std_cpu = T.zeros(1, self.act_dim)


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            #print("Soft clipping grads")

            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input
        x = F.selu(self.m1(self.fc1(x)))

        output, h = self.rnn(x, h)

        f = F.selu(self.m2(self.fc2(output)))
        if self.tanh:
            f = T.tanh(self.fc3(f))
        else:
            f = self.fc3(f)
        return f, h


    def sample_action(self, s):
        x, h = self.forward(s)
        return T.normal(x[0], T.exp(self.log_std_cpu)), h


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward((batch_states, None))

        # Calculate probabilities
        if self.to_gpu:
            log_std_batch = self.log_std_gpu.expand_as(action_means)
        else:
            log_std_batch = self.log_std_cpu.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


class RNN_PG_H(nn.Module):
    def __init__(self, env, hid_dim=64, memory_dim=64, n_temp=3, tanh=False):
        super(RNN_PG_H, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.tanh = tanh
        self.n_temp = n_temp

        #  A la Feature extractor
        self.fc1 = nn.Linear(self.obs_dim, self.memory_dim)
        self.m1 = nn.LayerNorm(self.memory_dim)
        self.fc2 = nn.Linear(self.memory_dim, self.memory_dim)
        self.m2 = nn.LayerNorm(self.memory_dim)

        # Two rnn layers
        self.rnn_1 = nn.LSTM(self.memory_dim, self.memory_dim, self.n_temp, batch_first=True)
        self.rnn_2 = nn.LSTM(self.memory_dim, self.memory_dim, self.n_temp, batch_first=True)

        # Mapping from RNN layers into actions
        self.fc3 = nn.Linear(self.memory_dim, self.act_dim)

        # Smaller than 1 std so that exploration is not that wild
        self.log_std_low = T.ones(1, self.memory_dim) * -1
        self.log_std_high = T.zeros(1, self.act_dim)


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, state = input

        if state is None:
            h_1, h_2 = None, None
        else:
            h_1, h_2 = state

        rnn_features = F.selu(self.m1(self.fc1(x)))
        rnn_features = F.selu(self.m2(self.fc2(rnn_features)))

        a_1, h_1 = self.rnn_1(rnn_features, h_1)

        output, h_2 = self.rnn_1(a_1, h_2)

        if self.tanh:
            a_2 = T.tanh(self.fc3(output))
        else:
            a_2 = self.fc3(output)

        return a_2, (h_1, h_2)


    def forward_low(self, input):
        x, state = input

        if state is None:
            h_1, h_2 = None, None
        else:
            h_1, h_2 = state

        rnn_features = F.selu(self.m1(self.fc1(x)))
        rnn_features = F.selu(self.m2(self.fc2(rnn_features)))

        a_1, h_1 = self.rnn_1(rnn_features, h_1)

        output, h_2 = self.rnn_1(a_1, h_2)

        if self.tanh:
            a_2 = T.tanh(self.fc3(output))
        else:
            a_2 = self.fc3(output)

        return a_1, a_2, (h_1, h_2)


    def sample_low(self, input):
        x, state = input

        if state is None:
            h_1, h_2 = None, None
        else:
            h_1, h_2 = state

        rnn_features = F.selu(self.m1(self.fc1(x)))
        rnn_features = F.selu(self.m2(self.fc2(rnn_features)))

        a_1, h_1 = self.rnn_1(rnn_features, h_1)
        a_1 = T.normal(a_1, T.exp(self.log_std_low))

        output, h_2 = self.rnn_1(a_1, h_2)

        if self.tanh:
            a_2 = T.tanh(self.fc3(output))
        else:
            a_2 = self.fc3(output)

        return a_1[0], a_2[0], (h_1, h_2)


    def sample_high(self, input):
        x, state = input

        if state is None:
            h_1, h_2 = None, None
        else:
            h_1, h_2 = state

        rnn_features = F.selu(self.m1(self.fc1(x)))
        rnn_features = F.selu(self.m2(self.fc2(rnn_features)))

        a_1, h_1 = self.rnn_1(rnn_features, h_1)
        output, h_2 = self.rnn_1(a_1, h_2)

        if self.tanh:
            a_2 = T.tanh(self.fc3(output))
        else:
            a_2 = self.fc3(output)

        a_2 = T.normal(a_2[0], T.exp(self.log_std_high))

        return a_2, (h_1, h_2)


    def log_probs_low(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _, _ = self.forward_low((batch_states, None))

        # Calculate probabilities
        log_std_batch = self.log_std_low.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


    def log_probs_high(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward((batch_states, None))

        # Calculate probabilities
        log_std_batch = self.log_std_high.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


class RNN_V3_AUX(nn.Module):
    def __init__(self, env, hid_dim=64, memory_dim=64, n_temp=3, n_classif=3, tanh=False, to_gpu=False):
        super(RNN_V3_AUX, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.n_classif = n_classif
        self.tanh = tanh
        self.to_gpu = to_gpu

        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.m1 = nn.LayerNorm(self.obs_dim)
        self.fc2 = nn.Linear(self.obs_dim + self.memory_dim, self.hid_dim)
        self.m2 = nn.LayerNorm(self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)
        self.fcc_1 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fcc_2 = nn.Linear(self.hid_dim, n_classif)

        if to_gpu:
            self.log_std_gpu = T.zeros(1, self.act_dim).cuda()
        else:
            self.log_std_cpu = T.zeros(1, self.act_dim)


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            #print("Soft clipping grads")

            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input
        rnn_features = F.selu(self.m1(self.fc1(x)))

        output, h = self.rnn(rnn_features, h)

        f = F.selu(self.m2(self.fc2(T.cat((output, x), 2))))
        if self.tanh:
            f = T.tanh(self.fc3(f))
        else:
            f = self.fc3(f)
        return f, h


    def classif(self, input):
        x, h = input
        rnn_features = F.selu(self.m1(self.fc1(x)))

        output, h = self.rnn(rnn_features, h)
        f = F.selu(self.m2(self.fc2(T.cat((output, x), 2))))
        f1 = F.selu(self.fcc_1(f))
        f2 = self.fcc_2(f1)

        return f2


    def sample_action(self, s):
        x, h = self.forward(s)
        return T.normal(x[0], T.exp(self.log_std_cpu)), h


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward((batch_states, None))

        # Calculate probabilities
        if self.to_gpu:
            log_std_batch = self.log_std_gpu.expand_as(action_means)
        else:
            log_std_batch = self.log_std_cpu.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


class RNN_CLASSIF_BASIC(nn.Module):
    def __init__(self, env, hid_dim=32, memory_dim=32, n_temp=3, n_classes=3, to_gpu=False, obs_dim=None):
        super(RNN_CLASSIF_BASIC, self).__init__()
        if env is None:
            self.obs_dim = obs_dim
        else:
            self.obs_dim = env.obs_dim
        self.act_dim = n_classes
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.n_classes = n_classes
        self.to_gpu = to_gpu

        self.m1 = nn.LayerNorm(self.obs_dim)

        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.fc2 = nn.Linear(self.memory_dim, self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input
        x = F.selu(self.m1(self.fc1(x)))

        output, h = self.rnn(x, h)

        x = F.selu(self.fc2(output))
        x = self.fc3(x)

        return x, h


class RNN_CLASSIF_ENV(nn.Module):
    def __init__(self, env, hid_dim=32, memory_dim=32, n_temp=3, n_classes=3, to_gpu=False, obs_dim=None):
        super(RNN_CLASSIF_ENV, self).__init__()
        if env is None:
            self.obs_dim = obs_dim
        else:
            self.obs_dim = env.obs_dim
        self.act_dim = n_classes
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.n_classes = n_classes
        self.to_gpu = to_gpu

        self.m1 = nn.LayerNorm(self.obs_dim)

        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.fc2 = nn.Linear(self.obs_dim + self.memory_dim, self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)

        T.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='leaky_relu')


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input
        x = F.selu(self.m1(self.fc1(x)))

        output, h = self.rnn(x, h)

        x = F.selu(self.fc2(T.cat((output, x), 2)))
        x = self.fc3(x)

        return x, h



class RNN_VAR_PG(nn.Module):
    def __init__(self, env, hid_dim=64, memory_dim=64, n_temp=3, tanh=False, to_gpu=False):
        super(RNN_VAR_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.tanh = tanh
        self.to_gpu = to_gpu

        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.fc2 = nn.Linear(self.obs_dim + self.memory_dim, self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)

        self.log_std_cpu = T.zeros(1, self.act_dim)
        #self.log_std_gpu = T.zeros(1, self.act_dim).cuda()


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            print("Soft clipping grads")

            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input
        x = F.selu(self.fc1(x))
        output, h = self.rnn(x, h)
        x = self.fc2(T.cat((output, x), 2))
        if self.tanh:
            x = T.tanh(self.fc3(x))
        else:
            x = self.fc3(x)
        return x, h


    def forward_batch(self, input):
        x_in, h = input
        lens = [len(el) for el in x_in]
        x = T.cat(x_in, 0)
        x = F.selu(self.fc1(x))
        x = T.split(x, lens)

        # Pack
        x_packed = T.nn.utils.rnn.pack_sequence(x)
        output_packed, h = self.rnn(x_packed, h)
        x_padded = T.nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)[0]
        x_list = T.unbind(x_padded, dim=0)
        x_list = [t[:l] for t,l in zip(x_list, lens)]
        x = T.cat(x_list)

        x = self.fc2(T.cat((x, T.cat(x_in)), 1))
        if self.tanh:
            x = T.tanh(self.fc3(x))
        else:
            x = self.fc3(x)
        return x, h


    def sample_action(self, s):
        x, h = self.forward(s)
        return T.normal(x[0], T.exp(self.log_std_cpu)), h


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward_batch((batch_states, None))

        # Calculate probabilities
        if self.to_gpu:
            log_std_batch = self.log_std_gpu.expand_as(action_means)
        else:
            log_std_batch = self.log_std_cpu.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(T.cat(batch_actions) - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class RNN_BLEND_PG(nn.Module):
    def __init__(self, env, hid_dim=64, memory_dim=24, n_temp=2, n_experts=4, tanh=False, to_gpu=False):
        super(RNN_BLEND_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.n_experts = n_experts
        self.memory_dim = memory_dim
        self.tanh = tanh
        self.to_gpu = to_gpu

        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc2 = nn.Linear(self.memory_dim, self.n_experts)

        self.fc_wl1_list = nn.ParameterList([])
        self.fc_bl1_list = nn.ParameterList([])
        self.fc_wl2_list = nn.ParameterList([])
        self.fc_bl2_list = nn.ParameterList([])
        self.fc_wl3_list = nn.ParameterList([])
        self.fc_bl3_list = nn.ParameterList([])

        for i in range(self.n_experts):
            # Make parameters
            w1 = nn.Parameter(data=T.zeros(self.hid_dim, self.obs_dim), requires_grad=True)
            b1 = nn.Parameter(data=T.randn(self.hid_dim) * 0.1, requires_grad=True)

            w2 = nn.Parameter(data=T.zeros(self.hid_dim, self.hid_dim), requires_grad=True)
            b2 = nn.Parameter(data=T.randn(self.hid_dim) * 0.1, requires_grad=True)

            w3 = nn.Parameter(data=T.zeros(self.act_dim, self.hid_dim), requires_grad=True)
            b3 = nn.Parameter(data=T.randn(self.act_dim) * 0.1, requires_grad=True)

            # Initialize parameters
            T.nn.init.xavier_uniform_(w1)
            #b1.data.fill_(0.01)

            T.nn.init.xavier_uniform_(w2)
            # b2.data.fill_(0.01)

            T.nn.init.xavier_uniform_(w3)
            # b3.data.fill_(0.01)

            # Add to parameter list
            self.fc_wl1_list.append(w1)
            self.fc_bl1_list.append(b1)

            self.fc_wl2_list.append(w2)
            self.fc_bl2_list.append(b2)

            self.fc_wl3_list.append(w3)
            self.fc_bl3_list.append(b3)

        self.log_std_cpu = T.zeros(1, self.act_dim)


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            print("Soft clipping grads")

            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input

        batch_dim = x.shape[0]
        seq_dim = x.shape[1]

        rnn_features = F.selu(self.fc1(x))
        rnn_output, h = self.rnn(rnn_features, h)
        coeffs = F.softmax(self.fc2(rnn_output), dim=2)
        output = T.zeros((x.shape[0], x.shape[1], self.act_dim))

        for b in range(batch_dim):
            for s in range(seq_dim):

                w1 = T.stack([coeffs[b, s, i] * self.fc_wl1_list[i] for i in range(self.n_experts)]).sum(0)
                b1 = T.stack([coeffs[b, s, i] * self.fc_bl1_list[i] for i in range(self.n_experts)]).sum(0)

                w2 = T.stack([coeffs[b, s, i] * self.fc_wl2_list[i] for i in range(self.n_experts)]).sum(0)
                b2 = T.stack([coeffs[b, s, i] * self.fc_bl2_list[i] for i in range(self.n_experts)]).sum(0)

                w3 = T.stack([coeffs[b, s, i] * self.fc_wl3_list[i] for i in range(self.n_experts)]).sum(0)
                b3 = T.stack([coeffs[b, s, i] * self.fc_bl3_list[i] for i in range(self.n_experts)]).sum(0)

                feat = F.selu(F.linear(x[b,s], w1, bias=b1))
                feat = F.linear(feat, w2, bias=b2)

                if self.tanh:
                    feat = T.tanh(F.linear(feat, w3, bias=b3))
                else:
                    feat = F.linear(feat, w3, bias=b3)

                output[b,s,:] = feat

            return output, h


    def sample_action(self, s):
        x, h = self.forward(s)
        return T.normal(x[0], T.exp(self.log_std_cpu)), h


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward((batch_states, None))

        # Calculate probabilities
        if self.to_gpu:
            log_std_batch = self.log_std_gpu.expand_as(action_means)
        else:
            log_std_batch = self.log_std_cpu.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


class RNN_BLEND_2_PG(nn.Module):
    def __init__(self, env, hid_dim=32, memory_dim=32, n_temp=3, n_experts=4, tanh=False, to_gpu=False):
        super(RNN_BLEND_2_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.n_experts = n_experts
        self.memory_dim = memory_dim
        self.tanh = tanh
        self.to_gpu = to_gpu

        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc2 = nn.Linear(self.memory_dim, self.n_experts)

        self.fc_1_exp_1 = nn.Linear(self.memory_dim, self.memory_dim)
        self.fc_2_exp_1 = nn.Linear(self.memory_dim, self.memory_dim)
        self.fc_3_exp_1 = nn.Linear(self.memory_dim, self.memory_dim)

        self.fc_1_exp_2 = nn.Linear(self.memory_dim, self.memory_dim)
        self.fc_2_exp_2 = nn.Linear(self.memory_dim, self.memory_dim)
        self.fc_3_exp_2 = nn.Linear(self.memory_dim, self.memory_dim)

        self.fc_1_exp_3 = nn.Linear(self.memory_dim, self.memory_dim)
        self.fc_2_exp_3 = nn.Linear(self.memory_dim, self.memory_dim)
        self.fc_3_exp_3 = nn.Linear(self.memory_dim, self.memory_dim)

        self.fc_1_exp_4 = nn.Linear(self.memory_dim, self.memory_dim)
        self.fc_2_exp_4 = nn.Linear(self.memory_dim, self.memory_dim)
        self.fc_3_exp_4 = nn.Linear(self.memory_dim, self.memory_dim)

        self.fc_exp_f = nn.Linear(self.memory_dim, self.act_dim)

        self.log_std_cpu = T.zeros(1, self.act_dim)


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            print("Soft clipping grads")

            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input

        rnn_features = F.selu(self.fc1(x))
        rnn_output, h = self.rnn(rnn_features, h)
        coeffs = F.softmax(self.fc2(rnn_output), dim=2)

        l1_cum = []
        l1_cum.append(self.fc_1_exp_1(rnn_output) * coeffs[:, :, 0:1].repeat(1, 1, self.memory_dim))
        l1_cum.append(self.fc_1_exp_2(rnn_output) * coeffs[:, :, 1:2].repeat(1, 1, self.memory_dim))
        l1_cum.append(self.fc_1_exp_3(rnn_output) * coeffs[:, :, 2:3].repeat(1, 1, self.memory_dim))
        l1_cum.append(self.fc_1_exp_4(rnn_output) * coeffs[:, :, 3:4].repeat(1, 1, self.memory_dim))
        l1 = F.selu(T.stack(l1_cum).sum(0, keepdim=False))

        l2_cum = []
        l2_cum.append(self.fc_2_exp_1(l1) * coeffs[:, :, 0:1].repeat(1, 1, self.memory_dim))
        l2_cum.append(self.fc_2_exp_2(l1) * coeffs[:, :, 1:2].repeat(1, 1, self.memory_dim))
        l2_cum.append(self.fc_2_exp_3(l1) * coeffs[:, :, 2:3].repeat(1, 1, self.memory_dim))
        l2_cum.append(self.fc_2_exp_4(l1) * coeffs[:, :, 3:4].repeat(1, 1, self.memory_dim))
        l2 = F.selu(T.stack(l2_cum).sum(0, keepdim=False))

        l3_cum = []
        l3_cum.append(self.fc_3_exp_1(l2) * coeffs[:, :, 0:1].repeat(1, 1, self.memory_dim))
        l3_cum.append(self.fc_3_exp_2(l2) * coeffs[:, :, 1:2].repeat(1, 1, self.memory_dim))
        l3_cum.append(self.fc_3_exp_3(l2) * coeffs[:, :, 2:3].repeat(1, 1, self.memory_dim))
        l3_cum.append(self.fc_3_exp_4(l2) * coeffs[:, :, 3:4].repeat(1, 1, self.memory_dim))
        l3 = F.selu(T.stack(l3_cum).sum(0, keepdim=False))

        if self.tanh:
            output = T.tanh(self.fc_exp_f(l3))
        else:
            output = self.fc_exp_f(l3)

        return output, h


    def sample_action(self, s):
        x, h = self.forward(s)
        return T.normal(x[0], T.exp(self.log_std_cpu)), h


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward((batch_states, None))

        # Calculate probabilities
        if self.to_gpu:
            log_std_batch = self.log_std_gpu.expand_as(action_means)
        else:
            log_std_batch = self.log_std_cpu.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


class RNN_BLEND_3_PG(nn.Module):
    def __init__(self, env, hid_dim=32, memory_dim=32, n_temp=3, n_experts=3, tanh=False, to_gpu=False):
        super(RNN_BLEND_3_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.n_experts = n_experts
        self.memory_dim = memory_dim
        self.tanh = tanh
        self.to_gpu = to_gpu

        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc2 = nn.Linear(self.memory_dim, self.n_experts)

        self.fc_exp_1 = [nn.Linear(self.memory_dim, self.memory_dim) for _ in range(n_experts)]
        self.fc_exp_2 = [nn.Linear(self.memory_dim, self.memory_dim) for _ in range(n_experts)]
        self.fc_exp_3 = [nn.Linear(self.memory_dim, self.memory_dim) for _ in range(n_experts)]
        self.fc_exp_f = nn.Linear(self.memory_dim, self.act_dim)

        self.log_std_cpu = T.zeros(1, self.act_dim)


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            print("Soft clipping grads")

            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input

        rnn_features = F.selu(self.fc1(x))
        rnn_output, h = self.rnn(rnn_features, h)
        coeffs = F.softmax(self.fc2(rnn_output), dim=2)
        selection = None

        l1 = T.stack([fc(rnn_output) * coeffs[:, :, i:i+1].repeat(1,1,self.memory_dim) for i, fc in enumerate(self.fc_exp_1)]).sum(0, keepdim=False)
        l2 = T.stack([fc(l1) * coeffs[:, :, i:i+1].repeat(1,1,self.memory_dim) for i, fc in enumerate(self.fc_exp_2)]).sum(0, keepdim=False)
        l3 = T.stack([fc(l2) * coeffs[:, :, i:i+1].repeat(1,1,self.memory_dim) for i, fc in enumerate(self.fc_exp_3)]).sum(0, keepdim=False)

        if self.tanh:
            output = T.tanh(self.fc_exp_f(l3))
        else:
            output = self.fc_exp_f(l3)

        return output, h


    def sample_action(self, s):
        x, h = self.forward(s)
        return T.normal(x[0], T.exp(self.log_std_cpu)), h


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward((batch_states, None))

        # Calculate probabilities
        if self.to_gpu:
            log_std_batch = self.log_std_gpu.expand_as(action_means)
        else:
            log_std_batch = self.log_std_cpu.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


class RNN_S(nn.Module):
    def __init__(self, env, hid_dim=48, memory_dim=24, tanh=False):
        super(RNN_S, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.tanh = tanh

        self.rnn = nn.LSTMCell(self.obs_dim, self.memory_dim)
        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.fc2 = nn.Linear(self.obs_dim + self.memory_dim, self.act_dim)

        self.log_std = T.zeros(1, self.act_dim)


    def clip_grads(self, bnd=1):
        self.rnn.weight_hh.grad.clamp_(-bnd, bnd)
        self.rnn.weight_ih.grad.clamp_(-bnd, bnd)
        self.rnn.bias_hh.grad.clamp_(-bnd, bnd)
        self.rnn.bias_ih.grad.clamp_(-bnd, bnd)

        self.fc1.weight.grad.clamp_(-bnd, bnd)
        self.fc1.bias.grad.clamp_(-bnd, bnd)
        self.fc2.weight.grad.clamp_(-bnd, bnd)
        self.fc2.bias.grad.clamp_(-bnd, bnd)


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            print("Soft clipping grads")
            self.rnn.weight_hh.grad = (self.rnn.weight_hh.grad / maxval) * bnd
            self.rnn.weight_ih.grad = (self.rnn.weight_ih.grad / maxval) * bnd
            self.rnn.bias_hh.grad = (self.rnn.bias_hh.grad / maxval) * bnd
            self.rnn.bias_ih.grad = (self.rnn.bias_ih.grad / maxval) * bnd
            self.fc1.weight.grad = (self.fc1.weight.grad / maxval) * bnd
            self.fc1.bias.grad = (self.fc1.bias.grad / maxval) * bnd
            self.fc2.weight.grad = (self.fc2.weight.grad / maxval) * bnd
            self.fc2.bias.grad = (self.fc2.bias.grad / maxval) * bnd


    def print_info(self):
        print("-------------------------------")
        print("w_hh", self.rnn.weight_hh.data.max(), self.rnn.weight_hh.data.min())
        print("w_ih", self.rnn.weight_ih.data.max(), self.rnn.weight_ih.data.min())
        print("b_hh", self.rnn.bias_hh.data.max(), self.rnn.bias_hh.data.min())
        print("b_ih", self.rnn.bias_ih.data.max(), self.rnn.bias_ih.data.min())
        print("w_fc1", self.fc1.weight.data.max(), self.fc1.weight.data.min())
        print("b_fc1", self.fc1.bias.data.max(), self.fc1.weight.data.min())
        print("w_fc2", self.fc2.weight.data.max(), self.fc2.weight.data.min())
        print("b_fc2", self.fc2.bias.data.max(), self.fc2.weight.data.min())
        print("---")
        print("w_hh grad", self.rnn.weight_hh.grad.max(), self.rnn.weight_hh.grad.min())
        print("w_ih grad", self.rnn.weight_ih.grad.max(), self.rnn.weight_ih.grad.min())
        print("b_hh grad", self.rnn.bias_hh.grad.max(), self.rnn.bias_hh.grad.min())
        print("b_ih grad", self.rnn.bias_ih.grad.max(), self.rnn.bias_ih.grad.min())
        print("w_fc1 grad", self.fc1.weight.grad.max(), self.fc1.weight.grad.min())
        print("b_fc1 grad", self.fc1.bias.grad.max(), self.fc1.bias.grad.min())
        print("w_fc2 grad", self.fc2.weight.grad.max(), self.fc2.weight.grad.min())
        print("b_fc2 grad", self.fc2.bias.grad.max(), self.fc2.bias.grad.min())
        print("-------------------------------")


    def forward(self, input):
        x, h = input
        x = F.selu(self.fc1(x))
        h_, c_ = self.rnn(x, h)
        if self.tanh:
            x = T.tanh(self.fc2(T.cat((h_, x), 1)))
        else:
            x = self.fc2(T.cat((h_, x), 1))
        return x, (h_, c_)


class RNN_ML(nn.Module):
    def __init__(self, env, hid_dim=128, memory_dim=128, tanh=False):
        super(RNN_ML, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.tanh = tanh

        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, 3, batch_first=True)
        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.fc2 = nn.Linear(self.obs_dim + self.memory_dim, self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            print("Soft clipping grads")

            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input
        x = F.selu(self.fc1(x))
        output, h = self.rnn(x, h)
        x = self.fc2(T.cat((output, x), 2))
        if self.tanh:
            x = T.tanh(self.fc3(x))
        else:
            x = self.fc3(x)
        return x, h


class RNN_CLASSIF(nn.Module):
    def __init__(self, env, n_classes, hid_dim=48, memory_dim=24):
        super(RNN_CLASSIF, self).__init__()
        self.obs_dim = env.obs_dim
        self.n_classes = n_classes
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim

        self.rnn = nn.LSTMCell(self.obs_dim, self.memory_dim)
        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.fc2 = nn.Linear(self.obs_dim + self.memory_dim, self.act_dim)


    def clip_grads(self, bnd=1):
        self.rnn.weight_hh.grad.clamp_(-bnd, bnd)
        self.rnn.weight_ih.grad.clamp_(-bnd, bnd)
        self.rnn.bias_hh.grad.clamp_(-bnd, bnd)
        self.rnn.bias_ih.grad.clamp_(-bnd, bnd)

        self.fc1.weight.grad.clamp_(-bnd, bnd)
        self.fc1.bias.grad.clamp_(-bnd, bnd)
        self.fc2.weight.grad.clamp_(-bnd, bnd)
        self.fc2.bias.grad.clamp_(-bnd, bnd)


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            print("Soft clipping grads")
            self.rnn.weight_hh.grad = (self.rnn.weight_hh.grad / maxval) * bnd
            self.rnn.weight_ih.grad = (self.rnn.weight_ih.grad / maxval) * bnd
            self.rnn.bias_hh.grad = (self.rnn.bias_hh.grad / maxval) * bnd
            self.rnn.bias_ih.grad = (self.rnn.bias_ih.grad / maxval) * bnd
            self.fc1.weight.grad = (self.fc1.weight.grad / maxval) * bnd
            self.fc1.bias.grad = (self.fc1.bias.grad / maxval) * bnd
            self.fc2.weight.grad = (self.fc2.weight.grad / maxval) * bnd
            self.fc2.bias.grad = (self.fc2.bias.grad / maxval) * bnd


    def print_info(self):
        print("-------------------------------")
        print("w_hh", self.rnn.weight_hh.data.max(), self.rnn.weight_hh.data.min())
        print("w_ih", self.rnn.weight_ih.data.max(), self.rnn.weight_ih.data.min())
        print("b_hh", self.rnn.bias_hh.data.max(), self.rnn.bias_hh.data.min())
        print("b_ih", self.rnn.bias_ih.data.max(), self.rnn.bias_ih.data.min())
        print("w_fc1", self.fc1.weight.data.max(), self.fc1.weight.data.min())
        print("b_fc1", self.fc1.bias.data.max(), self.fc1.weight.data.min())
        print("w_fc2", self.fc2.weight.data.max(), self.fc2.weight.data.min())
        print("b_fc2", self.fc2.bias.data.max(), self.fc2.weight.data.min())
        print("---")
        print("w_hh grad", self.rnn.weight_hh.grad.max(), self.rnn.weight_hh.grad.min())
        print("w_ih grad", self.rnn.weight_ih.grad.max(), self.rnn.weight_ih.grad.min())
        print("b_hh grad", self.rnn.bias_hh.grad.max(), self.rnn.bias_hh.grad.min())
        print("b_ih grad", self.rnn.bias_ih.grad.max(), self.rnn.bias_ih.grad.min())
        print("w_fc1 grad", self.fc1.weight.grad.max(), self.fc1.weight.grad.min())
        print("b_fc1 grad", self.fc1.bias.grad.max(), self.fc1.bias.grad.min())
        print("w_fc2 grad", self.fc2.weight.grad.max(), self.fc2.weight.grad.min())
        print("b_fc2 grad", self.fc2.bias.grad.max(), self.fc2.bias.grad.min())
        print("-------------------------------")


    def forward(self, input):
        x, h = input
        x = F.selu(self.fc1(x))
        h_, c_ = self.rnn(x, h)
        x = self.fc2(T.cat((h_, x), 1))
        return x, (h_, c_)


class RNN(nn.Module):
    def __init__(self, env):
        super(RNN, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = 8

        self.rnn = nn.RNNCell(self.hid_dim, self.hid_dim)
        self.fc1 = nn.Linear(self.obs_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.act_dim)


    def init_hidden(self):
        return T.zeros((1, self.hid_dim))


    def forward(self, input):
        x, h = input
        x = T.tanh(self.fc1(x))
        h_ = self.rnn(x, h)
        x = self.fc2(h_)
        return x, h_


class C_Linear(nn.Module):
    def __init__(self, env):
        super(C_Linear, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.fc1 = nn.Linear(self.obs_dim, self.act_dim)

        self.log_std = T.zeros(1, self.act_dim)


    def forward(self, x):
        return self.fc1(x)


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class C_MLP(nn.Module):
    def __init__(self, env):
        super(C_MLP, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.fc1 = nn.Linear(self.obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.act_dim)

        self.log_std = T.zeros(1, self.act_dim)


    def forward(self, x):
        x = T.tanh(self.fc1(x))
        x = T.tanh(self.fc2(x))
        x = T.tanh(self.fc3(x))
        return x


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class C_ConvPolicy8_CP(nn.Module):
    def __init__(self, env):
        super(C_ConvPolicy8_CP, self).__init__()
        self.N_links = 4
        self.act_dim = self.N_links * 6 - 2

        # rep conv
        self.conv_1 = nn.Conv1d(14, 4, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(8, 8, kernel_size=3, stride=1)
        self.conv_4 = nn.Conv1d(8, 8, kernel_size=2, stride=1)

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(14, 8, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(8, 8, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(8, 4, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(4, 4, kernel_size=3, stride=1, padding=1)
        self.deconv_3 = nn.ConvTranspose1d(4, 6, kernel_size=3, stride=1, padding=1)
        self.deconv_4 = nn.ConvTranspose1d(20, 6, kernel_size=3, stride=1, padding=1)

        self.upsample = nn.Upsample(size=4)

        self.afun = F.selu

        self.log_std = T.zeros(1, self.act_dim)

    def forward(self, x):
        # Batch dimension
        M = x.shape[0]

        # z, qw, qx, qy, qz [b,5]
        obs = x[:, :5]

        # xd, yd, xz, xangd, yangd, zangd [b, 6]
        obsd = x[:, 5 + self.N_links * 6 - 2: 5 + self.N_links * 6 - 2 + 6]

        # qw, qx, qy, qz, xd, yd [b, 6]
        ext_obs = T.cat((obs[:, 1:5], obsd[:, 0:2]), 1).unsqueeze(2)

        # Joints angles
        jl = T.cat((T.zeros(M, 2), x[:, 5:5 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((M, 6, -1))

        # Joint angle velocities
        jdl = T.cat(
            (T.zeros(M, 2), x[:, 5 + self.N_links * 6 - 2 + 6:5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2]), 1)
        jdlrs = jdl.view((M, 6, -1))

        # Contacts
        jcl = x[:, 5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2:]
        jclrs = jcl.view((M, 2, -1))

        ocat = T.cat((jlrs, jdlrs, jclrs), 1)  # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(ocat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c3 = self.afun(self.conv_3(fm_c2))
        fm_c4 = self.afun(self.conv_4(fm_c3))

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_c4, ext_obs), 1)))
        emb_2 = self.afun(self.conv_emb_2(emb_1))

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(emb_2))
        fm_dc2 = self.afun(self.deconv_2(fm_dc1))
        fm_dc3 = self.afun(self.deconv_3(fm_dc2))
        fm_upsampled = F.interpolate(fm_dc3, size=4)
        fm_dc4 = T.tanh(self.deconv_4(T.cat((fm_upsampled, ocat), 1)))

        acts = fm_dc4.squeeze(2).view((M, -1))

        return acts[:, 2:]


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class C_ConvPolicy14_CP(nn.Module):
    def __init__(self, env):
        super(C_ConvPolicy14_CP, self).__init__()
        self.N_links = 7

        self.act_dim = self.N_links * 6 - 2

        # rep conv
        self.conv_1 = nn.Conv1d(14, 6, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv1d(6, 8, kernel_size=3, stride=1)
        self.conv_3 = nn.Conv1d(8, 8, kernel_size=3, stride=1)

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(14, 10, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(10, 10, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(10, 6, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_3 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_4 = nn.ConvTranspose1d(20, 6, kernel_size=3, stride=1, padding=1)

        self.afun = F.selu

        self.log_std = T.zeros(1, self.act_dim)

    def forward(self, x):
        # Batch dimension
        M = x.shape[0]

        # z, qw, qx, qy, qz [b,5]
        obs = x[:, :5]

        # xd, yd, xz, xangd, yangd, zangd [b, 6]
        obsd = x[:, 5 + self.N_links * 6 - 2: 5 + self.N_links * 6 - 2 + 6]

        # qw, qx, qy, qz, xd, yd [b, 6]
        ext_obs = T.cat((obs[:, 1:5], obsd[:, 0:2]), 1).unsqueeze(2)

        # Joints angles
        jl = T.cat((T.zeros(M, 2), x[:, 5:5 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((M, 6, -1))

        # Joint angle velocities
        jdl = T.cat(
            (T.zeros(M, 2), x[:, 5 + self.N_links * 6 - 2 + 6:5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2]), 1)
        jdlrs = jdl.view((M, 6, -1))

        # Contacts
        jcl = x[:, 5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2:]
        jclrs = jcl.view((M, 2, -1))

        ocat = T.cat((jlrs, jdlrs, jclrs), 1)  # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(ocat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c3 = self.afun(self.conv_3(fm_c2))

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_c3, ext_obs),1)))
        emb_2 = self.afun(self.conv_emb_2(emb_1))

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(emb_2))
        fm_dc2 = self.afun(self.deconv_2(fm_dc1))
        fm_dc3 = self.afun(self.deconv_3(fm_dc2))
        fm_dc4 = T.tanh(self.deconv_4(T.cat((fm_dc3, ocat), 1)))

        acts = fm_dc4.squeeze(2).view((M, -1))

        return acts[:, 2:]


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class C_ConvPolicy30_CP(nn.Module):
    def __init__(self, env):
        super(C_ConvPolicy30_CP, self).__init__()
        self.N_links = 15
        self.act_dim = self.N_links * 6 - 2

        # rep conv
        self.conv_1 = nn.Conv1d(14, 6, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv1d(6, 8, kernel_size=3, stride=1)
        self.conv_3 = nn.Conv1d(8, 8, kernel_size=3, stride=1)
        self.downsample = nn.AdaptiveAvgPool1d(5)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(14, 10, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(10, 10, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(10, 6, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_3 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_4 = nn.ConvTranspose1d(20, 6, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=13)

        self.afun = F.selu

        self.log_std = T.zeros(1, self.act_dim)


    def forward(self, x):
        # Batch dimension
        M = x.shape[0]

        # z, qw, qx, qy, qz [b,5]
        obs = x[:, :5]

        # xd, yd, xz, xangd, yangd, zangd [b, 6]
        obsd = x[:, 5 + self.N_links * 6 - 2: 5 + self.N_links * 6 - 2 + 6]

        # qw, qx, qy, qz, xd, yd [b, 6]
        ext_obs = T.cat((obs[:, 1:5], obsd[:, 0:2]), 1).unsqueeze(2)

        # Joints angles
        jl = T.cat((T.zeros(M, 2), x[:, 5:5 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((M, 6, -1))

        # Joint angle velocities
        jdl = T.cat(
            (T.zeros(M, 2), x[:, 5 + self.N_links * 6 - 2 + 6:5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2]), 1)
        jdlrs = jdl.view((M, 6, -1))

        # Contacts
        jcl = x[:, 5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2:]
        jclrs = jcl.view((M, 2, -1))

        ocat = T.cat((jlrs, jdlrs, jclrs), 1)  # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(ocat))
        fm_c1_ds = self.downsample(fm_c1)
        fm_c2 = self.afun(self.conv_2(fm_c1_ds))
        fm_c3 = self.afun(self.conv_3(fm_c2))

        # Avg pool through link channels
        fm_links = self.pool(fm_c3) # (1, N, 1)

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_links, ext_obs),1)))
        emb_2 = self.afun(self.conv_emb_2(emb_1))

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(emb_2))
        fm_dc2 = self.afun(self.deconv_2(fm_dc1))
        fm_dc2_us = self.upsample(fm_dc2)
        fm_dc3 = self.afun(self.deconv_3(fm_dc2_us))
        fm_dc4 = T.tanh(self.deconv_4(T.cat((fm_dc3, ocat), 1))) # Change jcat to jlrs

        acts = fm_dc4.squeeze(2).view((M, -1))

        return acts[:, 2:]


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class C_ConvPolicy_Iter_CP(nn.Module):
    def __init__(self, env):
        super(C_ConvPolicy_Iter_CP, self).__init__()
        self.N_links = env.N_links
        self.act_dim = self.N_links * 6 - 2

        # rep conv
        self.conv_1 = nn.Conv1d(20, 6, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(6, 6, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(6, 6, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv1d(14, 6, kernel_size=3, stride=1, padding=1)

        self.afun = F.selu
        self.log_std = T.zeros(1, self.act_dim)

    def forward(self, x):
        # Batch dimension
        M = x.shape[0]

        # z, qw, qx, qy, qz [b,5]
        obs = x[:, :5]

        # xd, yd, xz, xangd, yangd, zangd [b, 6]
        obsd = x[:, 5 + self.N_links * 6 - 2: 5 + self.N_links * 6 - 2 + 6]

        # qw, qx, qy, qz, xd, yd [b, 6]
        ext_obs = T.cat((obs[:, 1:5], obsd[:, 0:2]), 1).unsqueeze(2)
        ext_obs_rep = ext_obs.repeat((1, 1, self.N_links))

        # Joints angles
        jl = T.cat((T.zeros(M, 2), x[:, 5:5 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((M, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(M, 2), x[:, 5 + self.N_links * 6 - 2 + 6:5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2]), 1)
        jdlrs = jdl.view((M, 6, -1))

        # Contacts
        jcl = x[:, 5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2:]
        jclrs = jcl.view((M, 2, -1))

        ocat = T.cat((jlrs, jdlrs, ext_obs_rep, jclrs), 1)  # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(ocat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c3 = self.afun(self.conv_3(fm_c2))
        fm_c4 = T.tanh(self.conv_4(T.cat((fm_c3, jlrs, jclrs), 1)))

        acts = fm_c4.squeeze(2).view((M, -1))

        return acts[:, 2:]


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class C_PhasePolicy_ES(nn.Module):
    def __init__(self, env):
        super(C_PhasePolicy_ES, self).__init__()
        self.N_links = env.N_links
        self.act_dim = self.N_links * 6 - 2

        # Set phase states
        self.reset()

        # Increment matrix which will be added to phases every step
        self.step_increment = T.ones(1, 6, self.N_links) * 0.1

        self.conv_obs = nn.Conv1d(20, 6, kernel_size=3, stride=1, padding=1)
        self.conv_phase = nn.Conv1d(6, 6, kernel_size=3, stride=1, padding=1)

        self.afun = T.tanh


    def step_phase(self):
        self.phases = T.fmod(self.phases + self.step_increment, 2 * np.pi)


    def modify_phase(self, mask):
        self.phases = T.fmod(self.phases + mask, 2 * np.pi)


    def reset(self):
        self.phases = T.randn(1, 6, self.N_links) * 0.1


    def forward(self, x):

        # Batch dimension
        M = x.shape[0]

        # z, qw, qx, qy, qz [b,5]
        obs = x[:, :5]

        # xd, yd, xz, xangd, yangd, zangd [b, 6]
        obsd = x[:, 5 + self.N_links * 6 - 2: 5 + self.N_links * 6 - 2 + 6]

        # qw, qx, qy, qz, xd, yd [b, 6]
        ext_obs = T.cat((obs[:, 1:5], obsd[:, 0:2]), 1).unsqueeze(2)
        ext_obs_rep = ext_obs.repeat((1, 1, self.N_links))

        # Joints angles
        jl = T.cat((T.zeros(M, 2), x[:, 5:5 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((M, 6, -1))

        # Joint angle velocities
        jdl = T.cat(
            (T.zeros(M, 2), x[:, 5 + self.N_links * 6 - 2 + 6:5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2]), 1)
        jdlrs = jdl.view((M, 6, -1))

        # Contacts
        jcl = x[:, 5 + self.N_links * 6 - 2 + 6 + self.N_links * 6 - 2:]
        jclrs = jcl.view((M, 2, -1))

        ocat = T.cat((jlrs, jdlrs, ext_obs_rep, jclrs), 1)  # Concatenate j and jd so that they are 2 parallel channels

        phase_fm = self.afun(self.conv_obs(ocat)) * 0.3
        phase_deltas = self.afun(self.conv_phase(phase_fm))

        self.modify_phase(phase_deltas)
        self.step_phase()

        # Phases directly translate into torques
        acts = T.sin(self.phases.view(M, self.act_dim + 2))

        return acts[:, 2:]


class CM_MLP(nn.Module):
    def __init__(self, obs_dim, act_dim, n_hid):
        super(CM_MLP, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_hid = n_hid

        self.fc1 = nn.Linear(self.obs_dim, self.n_hid)
        self.fc2 = nn.Linear(self.n_hid, self.n_hid)
        self.fc3 = nn.Linear(self.n_hid, self.act_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


class CM_RNN(nn.Module):
    def __init__(self, obs_dim, output_dim, n_hid):
        super(CM_RNN, self).__init__()

        self.n_hid = n_hid
        self.obs_dim = obs_dim
        self.output_dim = output_dim

        self.in1 = nn.Linear(self.obs_dim, self.n_hid)
        self.rnn = nn.GRUCell(self.n_hid, self.n_hid)
        self.out1 = nn.Linear(self.n_hid, self.n_hid)
        self.out2 = nn.Linear(self.n_hid, self.output_dim)


    def forward(self, x, h):
        x = T.tanh(self.in1(x))
        h_ = self.rnn(x, h)
        x = T.tanh(self.out1(h_))
        return self.out2(x), h_


    def average_grads(self, N):
        self.in1.weight.grad /= N
        self.in1.bias.grad /= N
        self.rnn.weight_hh.grad /= N
        self.rnn.weight_ih.grad /= N
        self.rnn.bias_hh.grad /= N
        self.rnn.bias_ih.grad /= N
        self.out1.weight.grad /= N
        self.out1.bias.grad /= N
        self.out2.weight.grad /= N
        self.out2.bias.grad /= N


    def reset(self, batchsize=1):
        return T.zeros(1, self.n_hid)


class CM_Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, n_hid):
        super(CM_Policy, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_hid = n_hid

        # Set states
        self.reset()

        self.rnn = nn.GRUCell(obs_dim, n_hid)
        self.out = nn.Linear(n_hid, act_dim)


    def forward(self, x, h):
        h_ = self.rnn(x, h)
        return self.out(h_), h_

    def average_grads(self, N):
        self.rnn.weight_hh.grad /= N
        self.rnn.weight_ih.grad /= N
        self.rnn.bias_hh.grad /= N
        self.rnn.bias_ih.grad /= N
        self.out.weight.grad /= N
        self.out.bias.grad /= N


    def reset(self, batchsize=1):
        return T.zeros(batchsize, self.n_hid).float()


class GYM_Linear(nn.Module):
    def __init__(self, env):
        super(GYM_Linear, self).__init__()
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.fc1 = nn.Linear(self.obs_dim, self.act_dim)


    def forward(self, x):
        return self.fc1(x)


class FB_RNN(nn.Module):
    def __init__(self, env):
        super(FB_RNN, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = 24


        self.rnn = nn.RNNCell(self.obs_dim, self.hid_dim)
        self.xp = nn.Linear(self.obs_dim, self.obs_dim)
        self.pa = nn.Linear(self.hid_dim, self.act_dim)
        self.ph = nn.Linear(self.hid_dim, self.hid_dim)
        self.ah = nn.Linear(self.act_dim, self.hid_dim)


    def init_hidden(self):
        return T.zeros((1, self.hid_dim))


    def forward(self, input):
        x, h = input

        # Input to rnn
        x = T.tanh(self.xp(x))

        # Pre-hidden state
        p = self.rnn(x, h)

        # Action output
        a = T.tanh(self.pa(p))

        # Next hidden state
        h_ = T.tanh(self.ph(p) + self.ah(a))

        return a, h_

    def wstats(self):
        return self.rnn.weight_ih.data.min(), self.rnn.weight_ih.data.max()


class NN_HEX(nn.Module):
    def __init__(self, env, hid_dim=64, tanh=False, std_fixed=True):
        super(NN_HEX, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.tanh = tanh

        self.fc_in_1 = nn.Linear(7, 6)
        self.fc_in_2 = nn.Linear(6, 3)

        self.fc_m_1 = nn.Linear(3 * 6 + 4, 24)
        self.fc_m_2 = nn.Linear(24, 3 * 6)

        self.fc_out_1 = nn.Linear(7 + 3, 6)
        self.fc_out_2 = nn.Linear(6, 3)

        if std_fixed:
            self.log_std = T.zeros(1, self.act_dim)
        else:
            self.log_std = nn.Parameter(T.zeros(1, self.act_dim))


    def forward(self, x):

        # Leg features
        l0 = T.cat([x[:, 0 * 3:0 * 3 + 3], x[:, 6 + 0 * 3: 6 + 0 * 3 + 3], x[:, 12 + 0:12 + 0 + 1]], 1)
        l1 = T.cat([x[:, 1 * 3:1 * 3 + 3], x[:, 6 + 1 * 3: 6 + 1 * 3 + 3], x[:, 12 + 1:12 + 1 + 1]], 1)
        l2 = T.cat([x[:, 2 * 3:2 * 3 + 3], x[:, 6 + 2 * 3: 6 + 2 * 3 + 3], x[:, 12 + 2:12 + 2 + 1]], 1)
        l3 = T.cat([x[:, 3 * 3:3 * 3 + 3], x[:, 6 + 3 * 3: 6 + 3 * 3 + 3], x[:, 12 + 3:12 + 3 + 1]], 1)
        l4 = T.cat([x[:, 4 * 3:4 * 3 + 3], x[:, 6 + 4 * 3: 6 + 4 * 3 + 3], x[:, 12 + 4:12 + 4 + 1]], 1)
        l5 = T.cat([x[:, 5 * 3:5 * 3 + 3], x[:, 6 + 5 * 3: 6 + 5 * 3 + 3], x[:, 12 + 5:12 + 5 + 1]], 1)

        in_f1_0 = F.relu(self.fc_in_1(l0))
        in_f1_1 = F.relu(self.fc_in_1(l1))
        in_f1_2 = F.relu(self.fc_in_1(l2))
        in_f1_3 = F.relu(self.fc_in_1(l3))
        in_f1_4 = F.relu(self.fc_in_1(l4))
        in_f1_5 = F.relu(self.fc_in_1(l5))

        in_f2_0 = F.relu(self.fc_in_2(in_f1_0))
        in_f2_1 = F.relu(self.fc_in_2(in_f1_1))
        in_f2_2 = F.relu(self.fc_in_2(in_f1_2))
        in_f2_3 = F.relu(self.fc_in_2(in_f1_3))
        in_f2_4 = F.relu(self.fc_in_2(in_f1_4))
        in_f2_5 = F.relu(self.fc_in_2(in_f1_5))

        in_cat = T.cat([x[:, 12:16], in_f2_0, in_f2_1, in_f2_2, in_f2_3, in_f2_4, in_f2_5], 1)

        m_1 = F.relu(self.fc_m_1(in_cat))
        m_2 = F.relu(self.fc_m_2(m_1))

        out_1_0 = F.relu(self.fc_out_1(T.cat([l0, m_2[:, 0 * 3:0 * 3 + 3]], 1)))
        out_1_1 = F.relu(self.fc_out_1(T.cat([l1, m_2[:, 1 * 3:1 * 3 + 3]], 1)))
        out_1_2 = F.relu(self.fc_out_1(T.cat([l2, m_2[:, 2 * 3:2 * 3 + 3]], 1)))
        out_1_3 = F.relu(self.fc_out_1(T.cat([l3, m_2[:, 3 * 3:3 * 3 + 3]], 1)))
        out_1_4 = F.relu(self.fc_out_1(T.cat([l4, m_2[:, 4 * 3:4 * 3 + 3]], 1)))
        out_1_5 = F.relu(self.fc_out_1(T.cat([l5, m_2[:, 5 * 3:5 * 3 + 3]], 1)))

        out_2_0 = self.fc_out_2(out_1_0)
        out_2_1 = self.fc_out_2(out_1_1)
        out_2_2 = self.fc_out_2(out_1_2)
        out_2_3 = self.fc_out_2(out_1_3)
        out_2_4 = self.fc_out_2(out_1_4)
        out_2_5 = self.fc_out_2(out_1_5)

        return T.cat([out_2_0, out_2_1, out_2_2, out_2_3, out_2_4, out_2_5], 1)


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            # print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class CYC_HEX(nn.Module):
    def __init__(self):
        super(CYC_HEX, self).__init__()

        self.phase_stepsize = 0.1
        self.phase_global = 0

        self.phase_scale_global = T.nn.Parameter(T.ones(1))
        self.phase_offset_joints = T.nn.Parameter(T.zeros(18))
        self.amplitude_offset_joints = T.nn.Parameter(T.zeros(9))
        self.amplitude_scale_joints = T.nn.Parameter(T.ones(3))


    def forward(self, _):
        amplitude_offset_joints_expanded = T.cat([self.amplitude_offset_joints[0:3].repeat(2),
                                                  self.amplitude_offset_joints[3:6].repeat(2),
                                                  self.amplitude_offset_joints[6:].repeat(2)])
        # amplitude_scale_joints_expanded = T.cat([self.amplitude_scale_joints[0:3].repeat(2),
        #                                           self.amplitude_scale_joints[3:6].repeat(2),
        #                                           self.amplitude_scale_joints[6:].repeat(2)])
        act = amplitude_offset_joints_expanded + self.amplitude_scale_joints.repeat(6) * T.sin(self.phase_global + self.phase_offset_joints).unsqueeze(0)
        self.phase_global = (self.phase_global + self.phase_stepsize * self.phase_scale_global) % (2 * np.pi)
        return act


class CYC_HEX_BS(nn.Module):
    def __init__(self):
        super(CYC_HEX_BS, self).__init__()

        self.phase_stepsize = 0.3
        self.phase_global = 0.0

        self.phase_scale_global = T.nn.Parameter(T.ones(1))
        #self.phase_offset_R = T.nn.Parameter(T.ones(1))
        self.phase_offset_joints = T.nn.Parameter(T.zeros(9))
        self.amplitude_offset_joints = T.nn.Parameter(T.zeros(9))
        self.amplitude_scale_joints = T.nn.Parameter(T.ones(9))

    def forward(self, _):
        phase_L = self.phase_global
        phase_R = (self.phase_global + np.pi) % (2 * np.pi)
        phase_LR_vec = T.tensor([phase_L, phase_L, phase_L, phase_R, phase_R, phase_R]).repeat(3)
        phase_offset_joints_expanded = T.cat([self.phase_offset_joints[0:3].repeat(2),
                                              self.phase_offset_joints[3:6].repeat(2),
                                              self.phase_offset_joints[6:].repeat(2)])
        amplitude_offset_joints_expanded = T.cat([self.amplitude_offset_joints[0:3].repeat(2),
                                                  self.amplitude_offset_joints[3:6].repeat(2),
                                                  self.amplitude_offset_joints[6:].repeat(2)])
        amplitude_scale_joints_expanded = T.cat([self.amplitude_scale_joints[0:3].repeat(2),
                                                  self.amplitude_scale_joints[3:6].repeat(2),
                                                  self.amplitude_scale_joints[6:].repeat(2)])

        act = amplitude_offset_joints_expanded + amplitude_scale_joints_expanded * T.sin(phase_LR_vec + phase_offset_joints_expanded).unsqueeze(0)
        self.phase_global = (self.phase_global + self.phase_stepsize * self.phase_scale_global) % (2 * np.pi)
        return act


class CYC_HEX_NN(nn.Module):
    def __init__(self, obs_dim):
        super(CYC_HEX_NN, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = 18 + 1
        self.hidden_dim = 8

        self.phase_stepsize = 0.1
        self.phase_global = 0

        self.f1 = nn.Linear(self.obs_dim, self.act_dim)
        #self.f2 = nn.Linear(self.hidden_dim, self.act_dim)

    def forward(self, x):
        #x = T.tensor([[1.,0.,0.,0.]])
        #x1 = T.tanh(self.f1(x))
        out = self.f1(x)

        act = T.sin(self.phase_global + out[:, :18])
        self.phase_global = (self.phase_global + self.phase_stepsize * (out[:, 18] + 1)) % (2 * np.pi)
        return act


class CYC_HEX_NN2(nn.Module):
    def __init__(self, obs_dim):
        super(CYC_HEX_NN2, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = 18 + 1
        self.hidden_dim = 8

        self.phase_stepsize = 0.3
        self.phase_global = 0

        self.phase_scale_global = T.ones(1,1)
        self.phase_offset_joints = T.zeros(1,18)

        self.l1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.act_dim)


    def forward(self, x):
        x1 = T.nn.tanh(self.l1(x))
        inc = self.l2(x1)

        self.phase_scale_global += inc[:, 18]
        self.phase_offset_joints += inc[:, :18]

        act = T.sin(self.phase_global + self.phase_offset_joints)
        self.phase_global = (self.phase_global + self.phase_stepsize * self.phase_scale_global) % (2 * np.pi)
        return act