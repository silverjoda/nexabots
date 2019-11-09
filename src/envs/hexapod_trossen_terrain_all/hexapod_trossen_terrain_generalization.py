import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
import cv2
import math
from math import sqrt, acos, fabs, ceil
from opensimplex import OpenSimplex


import random
import string
#
# import gym
# from gym import spaces
# from gym.utils import seeding

class Hexapod():
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "assets/hexapod_trossen_")

    def __init__(self, env_list=None, max_n_envs=3):
        print("Trossen hexapod envs: {}".format(env_list))

        if env_list is None:
            self.env_list = ["flat"]
        else:
            self.env_list = env_list

        self.ID = '_'.join(self.env_list)
        self.specific_env_len = 50
        self.env_scaling = self.specific_env_len / 38.

        self.modelpath = Hexapod.MODELPATH
        self.n_envs = np.minimum(max_n_envs, len(self.env_list))
        self.s_len = 500
        self.max_steps = int(self.n_envs * self.s_len * 0.7)
        self.env_change_prob = 0.2
        self.env_width = 20
        self.cumulative_environment_reward = None
        self.walls = True

        # self.joints_rads_low = np.array([-0.4, -1.2, -1.0] * 6)
        # self.joints_rads_high = np.array([0.4, 0.2, 0.6] * 6)
        self.joints_rads_low = np.array([-0.6, -1.4, -0.8] * 6)
        self.joints_rads_high = np.array([0.6, 0.4, 0.8] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        # Parameters to be varied
        self.variable_param_dict = {'friction' : False,
                                    'torso_mass' : False,
                                    'k_p': False,
                                    'armature': False,
                                    'tibia_lengths' : False}

        self.friction_range = [0.1, 1.0]
        self.torso_mass_range = [0.1, 5]
        self.k_p_range = [30, 80]
        self.armature_range = [1.5, 1.6]
        self.tibia_lengths_range = [0.08, 0.14]

        self.use_HF = False
        self.HF_width = 6
        self.HF_length = 20

        self.vel_sum = 0

        self.generate_hybrid_env(self.n_envs, self.max_steps)
        self.reset()

        #self.observation_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.obs_dim,))
        #self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.act_dim,))


    def setupcam(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * .3
        self.viewer.cam.lookat[0] = -0.1
        self.viewer.cam.lookat[1] = -1
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -30


    def scale_joints(self, joints):
        return joints
        #
        # sjoints = np.array(joints)
        # sjoints = ((sjoints - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1
        # return sjoints


    def scale_action(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low


    def scale_inc(self, action):
        action *= (self.joints_rads_diff / 2.)
        joint_list = np.array(self.sim.get_state().qpos.tolist()[7:7 + self.act_dim])
        joint_list += action
        ctrl = np.clip(joint_list, self.joints_rads_low, self.joints_rads_high)
        return ctrl


    def scale_torque(self, action):
        return action


    def get_obs(self):
        qpos = self.sim.get_state().qpos.tolist()
        qvel = self.sim.get_state().qvel.tolist()
        a = qpos + qvel
        return np.asarray(a, dtype=np.float32)


    def get_obs_dict(self):
        od = {}
        # Intrinsic parameters
        for j in self.sim.model.joint_names:
            od[j + "_pos"] = self.sim.data.get_joint_qpos(j)
            od[j + "_vel"] = self.sim.data.get_joint_qvel(j)

        # Contacts:
        od['contacts'] = (np.abs(np.array(self.sim.data.cfrc_ext[[4, 7, 10, 13, 16, 19]])).sum(axis=1) > 0.05).astype(np.float32)

        return od


    def get_state(self):
        return self.sim.get_state()


    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.q_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()


    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)

        self.viewer.render()


    def step(self, ctrl):
        ctrl = np.clip(ctrl, -1, 1)
        ctrl_pen = np.square(ctrl).mean()
        #torques = self.sim.data.actuator_force
        #ctrl_pen = np.square(torques).mean()

        ctrl = self.scale_action(ctrl)

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1
        #
        # if np.random.rand() < 0.002:
        #     self.model.geom_friction[0] = np.array([0.0001, 0.5, 0.5])
        #     print("Changed friction")

        obs = self.get_obs()

        # Angle deviation
        x, y, z, qw, qx, qy, qz = obs[:7]
        xd, yd, zd, thd, phid, psid = self.sim.get_state().qvel.tolist()[:6]
        #xa, ya, za, tha, phia, psia = self.sim.data.qacc.tolist()[:6]

        self.vel_sum += xd

        # Reward conditions
        target_vel = 0.4

        velocity_rew = 1. / (abs(xd - target_vel) + 1.) - 1. / (target_vel + 1.)
        velocity_rew = velocity_rew * (1/(1 + 30 * np.square(yd)))

        roll, pitch, _ = my_utils.quat_to_rpy([qw,qx,qy,qz])
        #yaw_deviation = np.min((abs((yaw % 6.183) - (0 % 6.183)), abs(yaw - 0)))

        q_yaw = 2 * acos(qw)

        r_neg = np.square(y) * 0.2 + \
                np.square(q_yaw) * 0.5 + \
                np.square(pitch) * 0.5 + \
                np.square(roll) * 0.5 + \
                ctrl_pen * 0.01 + \
                np.square(zd) * 0.7

        r_pos = velocity_rew * 6 # + (abs(self.prev_yaw_deviation) - abs(yaw_deviation)) * 3. + (abs(self.prev_y_deviation) - abs(y)) * 3.
        r = r_pos - r_neg

        # Reevaluate termination condition
        done = self.step_ctr > self.max_steps # or abs(y) > 0.3 or abs(yaw) > 0.6 or abs(roll) > 0.8 or abs(pitch) > 0.8
        contacts = (np.abs(np.array(self.sim.data.cfrc_ext[[4, 7, 10, 13, 16, 19]])).sum(axis=1) > 0.05).astype(np.float32)

        if self.use_HF:
            obs = np.concatenate([self.scale_joints(self.sim.get_state().qpos.tolist()[7:]),
                                  self.sim.get_state().qvel.tolist()[6:],
                                  self.sim.get_state().qvel.tolist()[:6],
                                  [qw, qx, qy, qz, y],
                                  contacts, self.get_local_hf(x,y).flatten()])
        else:
            obs = np.concatenate([self.scale_joints(self.sim.get_state().qpos.tolist()[7:]),
                                  self.sim.get_state().qvel.tolist()[6:],
                                  self.sim.get_state().qvel.tolist()[:6],
                                  [qw, qx, qy, qz, y],
                                  contacts])

        return obs, r, done, None


    def reset(self, init_pos = None):
        if np.random.rand() < self.env_change_prob:
            self.generate_hybrid_env(self.n_envs, self.specific_env_len * self.n_envs)
            time.sleep(0.3)

        self.viewer = None
        path = Hexapod.MODELPATH + "{}.xml".format(self.ID)

        while True:
            try:
                self.model = mujoco_py.load_model_from_path(path)
                break
            except Exception:
                pass

        self.sim = mujoco_py.MjSim(self.model)

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = 18 * 2 + 6 + 4 + 6 + 1
        self.act_dim = self.sim.data.actuator_length.shape[0]

        if self.use_HF:
            self.obs_dim += self.HF_width * self.HF_length

        # Reset env variables
        self.step_ctr = 0
        self.episodes = 0

        if self.use_HF:
            self.hf_data = self.model.hfield_data
            self.hf_ncol = self.model.hfield_ncol[0]
            self.hf_nrow = self.model.hfield_nrow[0]
            self.hf_column_meters = self.model.hfield_size[0][0] * 2
            self.hf_row_meters = self.model.hfield_size[0][1] * 2
            self.hf_grid = self.hf_data.reshape((self.hf_nrow, self.hf_ncol))
            self.hf_grid_aug = np.zeros((self.hf_nrow * 2, self.hf_ncol * 2))
            self.hf_grid_aug[int(self.hf_nrow / 2):self.hf_nrow + int(self.hf_nrow / 2),
            int(self.hf_ncol / 2):self.hf_ncol + int(self.hf_ncol / 2)] = self.hf_grid
            self.pixels_per_column = self.hf_ncol / float(self.hf_column_meters)
            self.pixels_per_row = self.hf_nrow / float(self.hf_row_meters)
            self.x_offset = 0.3
            self.y_offset = 0.6

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = 0.0 # np.random.rand() * 4 - 4
        init_q[1] = 0.0 # np.random.rand() * 8 - 4
        init_q[2] = 0.10
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        if init_pos is not None:
            init_q[0:3] += init_pos

        self.vel_sum = 0

        # Init_quat
        self.rnd_yaw = 0 # np.random.rand() * 2. - 1
        rnd_quat = my_utils.rpy_to_quat(0,0,self.rnd_yaw)
        init_q[3:7] = rnd_quat

        # Set environment state
        self.set_state(init_q, init_qvel)

        for i in range(30):
            self.sim.forward()
            self.sim.step()

        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs


    def get_local_hf(self, x, y):
        x_coord = int((x + self.x_offset) * self.pixels_per_column)
        y_coord = int((y + self.y_offset) * self.pixels_per_row)

        # Get heighfield patch
        patch = self.hf_grid_aug[self.hf_nrow + (y_coord - int(0.35 * self.pixels_per_row)):self.hf_nrow + y_coord + int(0.35 * self.pixels_per_row),
                self.hf_ncol + x_coord - int(0.4 * self.pixels_per_column):self.hf_ncol + x_coord + int(0.65 * self.pixels_per_column)]

        # Resize patch to correct dims
        patch_rs = cv2.resize(patch, (self.HF_length, self.HF_width), interpolation=cv2.INTER_NEAREST)
        return patch_rs


    def generate_hybrid_env(self, n_envs, steps):
        envs = np.random.choice(self.env_list, n_envs, replace=False)
        #envs = ["holes", "tiles", "holes"]

        if n_envs == 1:
            size_list = [steps]
            scaled_indeces_list = [0]
        else:
            size_list = []
            raw_indeces = np.linspace(0, 1, n_envs + 1)[1:-1]
            current_idx = 0
            scaled_indeces_list = []
            for idx in raw_indeces:
                idx_scaled = int(steps * idx) + np.random.randint(0, 70) - 40
                scaled_indeces_list.append(idx_scaled)
                size_list.append(idx_scaled - current_idx)
                current_idx = idx_scaled
            size_list.append(steps - int(steps * raw_indeces[-1]) + np.random.randint(0, 70) - 40)

        maplist = [self.generate_heightmap(m, s) for m, s in zip(envs, size_list)]
        total_hm = np.concatenate(maplist, 1)

        # Smoothen transitions
        bnd = 4
        if self.n_envs > 1:
            for s in scaled_indeces_list:
                total_hm_copy = np.array(total_hm)
                for i in range(s - bnd, s + bnd):
                    total_hm_copy[:, i] = np.mean(total_hm[:, i - bnd:i + bnd], axis=1)
                total_hm = total_hm_copy

        if self.walls:
            total_hm[0, :] = 255
            total_hm[:, 0] = 255
            total_hm[-1, :] = 255
            total_hm[:, -1] = 255
        else:
            total_hm[0, 0] = 255

        cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "assets/{}.png".format(self.ID)), total_hm)

        with open(Hexapod.MODELPATH + "template.xml", "r") as in_file:
            buf = in_file.readlines()

        with open(Hexapod.MODELPATH + self.ID + ".xml", "w") as out_file:
            for line in buf:
                if line.startswith('    <hfield name="hill"'):
                    out_file.write('    <hfield name="hill" file="{}.png" size="{} 0.6 0.6 0.1" /> \n '.format(self.ID, self.env_scaling * self.n_envs))
                elif line.startswith('    <geom name="floor" conaffinity="1" condim="3"'):
                    out_file.write('    <geom name="floor" conaffinity="1" condim="3" material="MatPlane" pos="{} 0 -.5" rgba="0.8 0.9 0.8 1" type="hfield" hfield="hill"/>'.format(self.env_scaling * self.n_envs * 0.6))
                elif line.startswith('    <joint armature=') and self.variable_param_dict['armature']:
                    out_file.write('    <joint armature="{}" damping="3" limited="true"/>'.format(
                        self.sample_from_range(self.armature_range)))
                elif line.startswith('      <geom name="torso_geom"') and self.variable_param_dict['torso_mass']:
                    out_file.write(
                        '      <geom name="torso_geom" pos="0 0 0" size="0.12 0.06 0.02" mass="{}" type="ellipsoid" material="MatHex"/>'.format(
                            self.sample_from_range(self.torso_mass_range)))
                elif line.startswith('    <geom conaffinity="0" condim="3" ') and self.variable_param_dict['friction']:
                    out_file.write(
                        '    <geom conaffinity="0" condim="3" density="5.0" friction="{} 0.5 0.5" margin="0.0001" rgba="0.2 0.2 0.2 0.8"/>'.format(
                            self.sample_from_range(self.friction_range)))
                elif line.startswith('    <position joint=') and self.variable_param_dict['k_p']:
                    out_file.write(line.replace('kp="40"', 'kp="{}"'.format(self.sample_from_range(self.k_p_range))))
                elif line.startswith('                <geom fromto="0.0 0.0 0.0 0.0 0.0 -0.095" name="tibia_') and \
                        self.variable_param_dict['tibia_lengths']:
                    out_file.write(line.replace('0.095', '{}'.format(self.sample_from_range(self.tibia_lengths_range))))

                else:
                    out_file.write(line)

        return envs, size_list, scaled_indeces_list


    def sample_from_range(self, range):
        return np.random.rand() * (range[1] - range[0]) + range[0]


    def generate_heightmap(self, env_name, env_length):
        hm = np.ones((self.env_width, env_length)) * 127

        if env_name == "flat":
            pass

        if env_name == "tiles":
            hm = np.random.randint(0, 30,
                                   size=(self.env_width // 3, env_length // int(3)),
                                   dtype=np.uint8).repeat(3, axis=0).repeat(int(12), axis=1) + 127

        if env_name == "pipe":
            pipe = np.ones((self.env_width, env_length))
            hm = pipe * np.expand_dims(np.square(np.linspace(-13, 13, self.env_width)), 0).T + 127

        if env_name == "holes":
            hm = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/holes1.png"))
            h, w, _ = hm.shape
            patch_y = 14
            patch_x = int(14 * self.s_len / 150.)
            rnd_h = np.random.randint(0, h - patch_x)
            rnd_w = np.random.randint(0, w - patch_y)
            hm = hm[rnd_w:rnd_w + patch_y, rnd_h:rnd_h + patch_x]
            hm = np.mean(hm, axis=2)
            hm = hm * 1.0 + 255 * 0.3
            hm = cv2.resize(hm, dsize=(env_length, self.env_width), interpolation=cv2.INTER_CUBIC) / 2.

        if env_name == "inverseholes":
            hm = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/holes1.png"))
            h, w, _ = hm.shape
            patchsize = 10
            while True:
                rnd_h = np.random.randint(0, h - patchsize)
                rnd_w = np.random.randint(0, w - patchsize)
                hm_tmp = hm[rnd_w:rnd_w + patchsize, rnd_h:rnd_h + patchsize]
                #assert hm.shape == (10,10,3)
                if np.min(hm_tmp[:, :2, :]) > 160: break

            hm = np.mean(hm_tmp, axis=2)
            hm = cv2.resize(hm, dsize=(env_length, self.env_width), interpolation=cv2.INTER_CUBIC)
            hm = 255 - hm
            hm *= 0.5
            hm += 127

        if env_name == "bumps":
            hm = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/bumps2.png"))
            h, w, _ = hm.shape
            patchsize = 50
            rnd_h = np.random.randint(0, h - patchsize)
            rnd_w = np.random.randint(0, w - patchsize)
            hm = hm[rnd_w:rnd_w + patchsize, rnd_h:rnd_h + patchsize]
            hm = np.mean(hm, axis=2)
            hm = cv2.resize(hm, dsize=(env_length, self.env_width), interpolation=cv2.INTER_CUBIC) / 2. + 127

        if env_name == "stairs":
            stair_height = 45
            stair_width = 4
            current_height = 0
            initial_offset = 15

            hm[:, :initial_offset] = 0

            for i in range(20):
                hm[:, initial_offset  + i * stair_width: initial_offset  + i * stair_width + stair_width] = current_height
                current_height += stair_height

            # for i in range(3):
            #     hm[:, 80 + i * stair_width:  80 + i * stair_width + stair_width] = current_height
            #
            # for i in range(4):
            #     hm[:, 60 + i * stair_width: 60 + i * stair_width + stair_width] = current_height
            #     current_height -= stair_height
            #
            #


        if env_name == "verts":
            wdiv = 4
            ldiv = 14
            hm = np.random.randint(0, 75,
                                   size=(self.env_width // wdiv, env_length // ldiv),
                                   dtype=np.uint8).repeat(wdiv, axis=0).repeat(ldiv, axis=1)
            hm[:, :50] = 0
            hm[hm < 50] = 0
            hm = 75 - hm


        if env_name == "triangles":
            cw = 10
            # Make even dimensions
            M = math.ceil(self.env_width)
            N = math.ceil(env_length)
            hm = np.zeros((M, N), dtype=np.float32)
            M_2 = math.ceil(M / 2)

            # Amount of 'tiles'
            Mt = 2
            Nt = int(env_length / 10.)
            obstacle_height = 0.25
            grad_mat = np.linspace(0, 1, cw)[:, np.newaxis].repeat(cw, 1)
            template_1 = np.ones((cw, cw)) * grad_mat * grad_mat.T * obstacle_height
            template_2 = np.ones((cw, cw)) * grad_mat * obstacle_height

            for i in range(Nt):
                if np.random.choice([True, False]):
                    hm[M_2 - cw: M_2, i * cw: i * cw + cw] = np.rot90(template_1, np.random.randint(0, 4))
                else:
                    hm[M_2 - cw: M_2, i * cw: i * cw + cw] = np.rot90(template_2, np.random.randint(0, 4))

                if np.random.choice([True, False]):
                    hm[M_2:M_2 + cw:, i * cw: i * cw + cw] = np.rot90(template_1, np.random.randint(0, 4))
                else:
                    hm[M_2:M_2 + cw:, i * cw: i * cw + cw] = np.rot90(template_2, np.random.randint(0, 4))

            # Walls
            hm[0, 0] = 1.

            # Multiply to full image resolution
            hm *= 255


        if env_name == "perlin":
            oSim = OpenSimplex(seed=int(time.time()))

            height = 120

            M = math.ceil(self.env_width)
            N = math.ceil(env_length)
            hm = np.zeros((M, N), dtype=np.float32)

            scale_x = 20
            scale_y = 20
            octaves = 4  # np.random.randint(1, 5)
            persistence = 1
            lacunarity = 2

            for i in range(M):
                for j in range(N):
                    for o in range(octaves):
                        sx = scale_x * (1 / (lacunarity ** o))
                        sy = scale_y * (1 / (lacunarity ** o))
                        amp = persistence ** o
                        hm[i][j] += oSim.noise2d(i / sx, j / sy) * amp

            wmin, wmax = hm.min(), hm.max()
            hm = (hm - wmin) / (wmax - wmin) * height


        return hm


    def demo(self):
        self.reset()

        for i in range(1000):
            #self.step(np.random.randn(self.act_dim))
            for i in range(100):
                self.step(np.zeros((self.act_dim)))
                self.render()
            for i in range(100):
                self.step(np.array([0., -1., 1.] * 6))
                self.render()
            for i in range(100):
                self.step(np.ones((self.act_dim)) * 1)
                self.render()
            for i in range(100):
                self.step(np.ones((self.act_dim)) * -1)
                self.render()


    def info(self):
        self.reset()
        for i in range(100):
            a = np.ones((self.act_dim)) * 0
            obs, _, _, _ = self.step(a)
            print(obs[[3, 4, 5]])
            self.render()
            time.sleep(0.01)

        print("-------------------------------------------")
        print("-------------------------------------------")


    def test_record(self, policy, ID):
        episode_states = []
        episode_acts = []
        for i in range(10):
            s = self.reset()
            cr = 0

            states = []
            acts = []

            for j in range(self.max_steps):
                states.append(s)
                action = policy(my_utils.to_tensor(s, True)).detach()[0].numpy()
                acts.append(action)
                s, r, done, od, = self.step(action)
                cr += r

            episode_states.append(np.concatenate(states))
            episode_acts.append(np.concatenate(acts))

            print("Total episode reward: {}".format(cr))

        np_states = np.concatenate(episode_states)
        np_acts = np.concatenate(episode_acts)

        np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "data/{}_states.npy".format(ID)) , np_states)
        np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "data/{}_acts.npy".format(ID)), np_acts)


    def test(self, policy, render=True):
        self.env_change_prob = 1
        N = 30
        rew = 0
        for i in range(N):
            obs = self.reset()
            cr = 0
            for j in range(int(self.max_steps)):
                action = policy(my_utils.to_tensor(obs, True)).detach()
                obs, r, done, od, = self.step(action[0].numpy())
                cr += r
                rew += r
                time.sleep(0.000)
                if render:
                    self.render()
            print("Total episode reward: {}".format(cr))
        print("Total average reward = {}".format(rew / N))


    def test_recurrent(self, policy):
        self.env_change_prob = 1
        self.reset()
        h_episodes = []
        N = 20
        rew = 0
        for i in range(N):
            h_list = []
            obs = self.reset()
            h = None
            cr = 0
            for j in range(self.max_steps):
                action, h = policy((my_utils.to_tensor(obs, True).unsqueeze(0), h))
                obs, r, done, od, = self.step(action[0].detach().numpy())
                cr += r
                rew += r
                time.sleep(0.001)
                self.render()
                #h_list.append(h[0][:,0,:].detach().numpy())
            print("Total episode reward: {}".format(cr))
            #h_arr = np.stack(h_list)
            #h_episodes.append(h_arr)

        print("Total average reward = {}".format(rew / N))
        exit()

        h_episodes_arr = np.stack(h_episodes)

        # Save hidden states
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "data/{}_states.npy".format(self.env_name))
        #np.save(filename, h_episodes_arr)


    def test_adapt(self, p1, p2, ID):
        self.env_list = ["flatpipe"]

        episode_states = []
        episode_acts = []
        ctr = 0
        while ctr < 1000:
            print("Iter: {}".format(ctr))
            current_policy_name = "p1"
            rnd_x = - 0.1 + np.random.rand() * 0.3 + np.random.randint(0,2) * 1.2
            s = self.reset(init_pos = np.array([rnd_x, 0, 0]))
            cr = 0
            states = []
            acts = []

            policy = p1

            for j in range(self.max_steps):
                x = self.sim.get_state().qpos.tolist()[0]

                if 2.2 > x > 0.8 and current_policy_name == "p1":
                    policy = p2
                    current_policy_name = "p2"
                    print("Policy switched to p2")

                if not (2.2 > x > 0.8) and current_policy_name == "p2":
                    policy = p1
                    current_policy_name = "p1"
                    print("Policy switched to p1")

                states.append(s)
                action = policy(my_utils.to_tensor(s, True)).detach()[0].numpy()
                acts.append(action)
                s, r, done, od, = self.step(action)
                cr += r

                #self.render()

            if cr < 50:
                continue
            ctr += 1

            episode_states.append(np.stack(states))
            episode_acts.append(np.stack(acts))

            print("Total episode reward: {}".format(cr))

        np_states = np.stack(episode_states)
        np_acts = np.stack(episode_acts)

        np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "data/states_{}.npy".format(ID)), np_states)
        np.save(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "data/acts_{}.npy".format(ID)), np_acts)


    def test_record_hidden(self, policy):
            self.reset()
            h_episodes = []
            for i in range(10):
                h_list = []
                obs = self.reset()
                h = None
                cr = 0
                for j in range(self.max_steps  * 2):
                    action, h = policy((my_utils.to_tensor(obs, True), h))
                    obs, r, done, od, = self.step(action[0].detach().numpy())
                    cr += r
                    time.sleep(0.001)
                    self.render()
                    h_list.append(h[0].detach().numpy())
                print("Total episode reward: {}".format(cr))
                h_arr = np.concatenate(h_list)
                h_episodes.append(h_arr)

            h_episodes_arr = np.stack(h_episodes)

            # Save hidden states
            filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "data/{}_states.npy".format(self.env_name))
            np.save(filename, h_episodes_arr)



if __name__ == "__main__":
    ant = Hexapod()
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()