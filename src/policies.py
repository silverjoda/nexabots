import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np

class Baseline(nn.Module):
    def __init__(self, N):
        super(Baseline, self).__init__()
        self.N_links = int(N / 2)
        self.fc1 = nn.Linear(93, 40)

    def forward(self, x):
        x = self.fc1(x)
        return x


class NN(nn.Module):
    def __init__(self, env):
        super(NN, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.fc1 = nn.Linear(self.obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.act_dim)

        #self.log_std = nn.Parameter(T.zeros(1, self.act_dim))
        self.log_std = T.zeros(1, self.act_dim)


    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)

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
       # fm_c1_ds = self.downsample(fm_c1)
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
    def __init__(self, N):
        super(StatePolicy, self).__init__()
        self.N_links = int(N / 2)

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


class ConvPolicy8_PG(nn.Module):
    def __init__(self, env):
        super(ConvPolicy8_PG, self).__init__()
        self.N_links = 4
        self.act_dim = self.N_links * 6 - 2

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

        self.upsample = nn.Upsample(size=4)

        self.afun = F.selu

        self.log_std = T.zeros(1, self.act_dim)

    def forward(self, x):
        N = x.shape[0]
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]

        # (psi, psid)
        ext_obs = T.cat((obs[:, 3:7], obsd[:, -1:]), 1)

        # Joints angles
        jl = T.cat((T.zeros(N, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((N, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(N, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((N, 6, -1))

        jcat = T.cat((jlrs, jdlrs), 1)  # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(jcat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c3 = self.afun(self.conv_3(fm_c2))
        fm_c4 = self.afun(self.conv_4(fm_c3))

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_c4, ext_obs.unsqueeze(2)), 1)))
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
       # fm_c1_ds = self.downsample(fm_c1)
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


class NN_PG(nn.Module):
    def __init__(self, env):
        super(NN_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.fc1 = nn.Linear(self.obs_dim, 64)
        #self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        #self.bn2 = nn.BatchNorm2d(64)
        self.fc3 = nn.Linear(64, self.act_dim)

        #self.log_std = nn.Parameter(T.zeros(1, self.act_dim))
        self.log_std = T.zeros(1, self.act_dim)


    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)

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
