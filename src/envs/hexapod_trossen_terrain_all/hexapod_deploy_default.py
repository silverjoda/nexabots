import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
import cv2
import math
from math import sqrt, acos, fabs, ceil
from opensimplex import OpenSimplex
import gym
from gym import spaces


class Hexapod(gym.Env):
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "assets/hexapod_trossen_")

    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, env_list=None, max_n_envs=3, specific_env_len=30, s_len=100, walls=True, target_vel=0.2, use_contacts=False, turn_dir=None):
        print("Trossen hexapod envs: {}".format(env_list))

        if env_list is None:
            self.env_list = ["flat"]
        else:
            self.env_list = env_list

        self.ID = '_'.join(self.env_list)
        self.specific_env_len = specific_env_len
        self.env_scaling = self.specific_env_len / 38.

        self.modelpath = Hexapod.MODELPATH
        self.n_envs = np.minimum(max_n_envs, len(self.env_list))
        self.s_len = s_len
        self.use_contacts = use_contacts
        self.max_steps = int(self.n_envs * self.s_len * 1.0)
        self.env_change_prob = 0.1
        self.env_width = 40
        self.current_difficulty = 0.3 # [0,1]
        self.difficulty_per_step_increment = 2e-7
        self.cumulative_environment_reward = None
        self.walls = walls
        self.turn_dir = turn_dir

        self.rnd_init_yaw = True
        self.replace_envs = True

        self.joints_rads_low = np.array([-0.3, -1.6, 0.7] * 6)
        self.joints_rads_high = np.array([0.3, 0.0, 1.9] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.use_HF = False
        self.HF_width = 6
        self.HF_length = 20
        self.target_vel = target_vel

        self.xd_queue = []

        self.generate_hybrid_env(self.n_envs, self.specific_env_len * self.n_envs)
        self.reset()

        self.observation_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.obs_dim,))
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.act_dim,))


    def setupcam(self):
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.distance = self.model.stat.extent * .3
        self.viewer.cam.lookat[0] = 2.
        self.viewer.cam.lookat[1] = 0.3
        self.viewer.cam.lookat[2] = 0.9
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = -10


    def scale_joints(self, joints):
        sjoints = np.array(joints)
        sjoints = ((sjoints - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1
        return sjoints


    def scale_action(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low


    def scale_inc(self, action):
        action *= (self.joints_rads_diff / 2.)
        joint_list = np.array(self.sim.get_state().qpos.tolist()[7:7 + self.act_dim])
        joint_list += action
        ctrl = np.clip(joint_list, self.joints_rads_low, self.joints_rads_high)
        return ctrl


    def get_obs(self):
        qpos = self.sim.get_state().qpos.tolist()
        qvel = self.sim.get_state().qvel.tolist()
        a = qpos + qvel
        return np.asarray(a, dtype=np.float32)


    def get_state(self):
        return self.sim.get_state()


    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.q_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()


    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()


    def draw(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()

    def myrender(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()


    def step(self, ctrl, render=False):
        ctrl = np.clip(ctrl, -1, 1)
        ctrl = self.scale_action(ctrl)
        self.sim.data.ctrl[:] = ctrl

        self.model.opt.timestep = 0.003
        for i in range(27):
            self.sim.forward()
            self.sim.step()
            if render:
                self.render()

        # Simulate read delay
        self.model.opt.timestep = 0.0008
        joints = []
        for i in range(18):
            joints.append(self.sim.get_state().qpos.tolist()[7 + i])
            self.sim.forward()
            self.sim.step()

        self.step_ctr += 1
        self.current_difficulty = np.minimum(self.current_difficulty + self.difficulty_per_step_increment, 1)
        torques = self.sim.data.actuator_force
        ctrl_pen = np.square(torques).mean()

        obs = self.get_obs()

        # Angle deviation
        x, y, z, qw, qx, qy, qz = obs[:7]
        xd, yd, zd, thd, phid, psid = self.sim.get_state().qvel.tolist()[:6]

        self.xd_queue.append(xd)
        if len(self.xd_queue) > 15:
            self.xd_queue.pop(0)
        xd_av = sum(self.xd_queue) / len(self.xd_queue)

        velocity_rew = 1. / (abs(xd_av - self.target_vel) + 1.) - 1. / (self.target_vel + 1.)
        velocity_rew *= (0.3 / self.target_vel)

        roll, pitch, _ = my_utils.quat_to_rpy([qw,qx,qy,qz])
        q_yaw = 2 * acos(qw)

        yaw_deviation = np.min((abs((q_yaw % 6.183) - (0 % 6.183)), abs(q_yaw - 0)))

        r_neg = np.square(q_yaw) * 0.5 + \
                np.square(pitch) * 0.5 + \
                np.square(roll) * 0.5 + \
                ctrl_pen * 0.000001 + \
                np.square(zd) * 0.7

        if self.env_list == ["flat"]:
            r_neg += ctrl_pen * 0.01

        r_correction = np.clip(abs(self.prev_deviation) - abs(yaw_deviation), -1, 1)
        r_pos = velocity_rew * 7 + r_correction * 15
        r = np.clip(r_pos - r_neg, -3, 3)

        if self.env_list == ["stairs"] or self.env_list == ["stairs_down"]:
            velocity_rew = 1. / (abs(xd_av - self.target_vel * 0.7) + 1.) - 1. / (self.target_vel * 0.7 + 1.)
            velocity_rew *= (0.3 / (self.target_vel * 0.7))

            r_neg = np.square(q_yaw) * 0.5 + \
                    np.square(pitch) * 0 + \
                    np.square(roll) * 0.2 + \
                    ctrl_pen * 0.00000 + \
                    np.square(zd) * 0

            r_correction = np.clip(abs(self.prev_deviation) - abs(yaw_deviation), -1, 1)
            r_pos = velocity_rew * 7 + r_correction * 15
            r = np.clip(r_pos - r_neg, -3, 3)

        if self.turn_dir is not None:
            r = np.clip(psid * (1 - 2 * (self.turn_dir == "LEFT")), -3, 3)

        self.prev_deviation = yaw_deviation

        # Reevaluate termination condition
        done = self.step_ctr > self.max_steps
        contacts = (np.abs(np.array(self.sim.data.sensordata[0:6], dtype=np.float32)) > 0.05).astype(np.float32) * 2 - 1.

        # See if changing contacts affects policy
        #np.random.randint(0,2, size=6) * 2 - 1
        #contacts = -np.ones(6)

        #clipped_torques = np.clip(torques * 0.05, -1, 1)
        c_obs = self.scale_joints(joints)
        quat = obs[3:7]

        if self.use_contacts:
            c_obs = np.concatenate([c_obs, contacts])

        c_obs = np.concatenate([c_obs, quat])
        return c_obs, r, done, {}


    def step_raw(self, ctrl):
        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()


    def reset(self, init_pos = None):
        if np.random.rand() < self.env_change_prob:
            self.generate_hybrid_env(self.n_envs, self.specific_env_len * self.n_envs)
            time.sleep(0.1)

        self.viewer = None
        path = Hexapod.MODELPATH + "{}.xml".format(self.ID)

        #
        # with open(Hexapod.MODELPATH + "limited.xml", "r") as in_file:
        #     buf = in_file.readlines()
        #
        # with open(Hexapod.MODELPATH + self.ID + ".xml", "w") as out_file:
        #     for line in buf:
        #         if line.startswith('    <position joint='):
        #             out_file.write('    <hfield name="hill" file="{}.png" size="{} 1.2 {} 0.1" /> \n '.format(self.ID, self.env_scaling * self.n_envs, 0.6 * height_SF))
        #         else:
        #             out_file.write(line)

        while True:
            try:
                self.model = mujoco_py.load_model_from_path(path)
                break
            except Exception:
                pass

        self.sim = mujoco_py.MjSim(self.model)
        self.model.opt.timestep = 0.003

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = 22 + self.use_contacts * 6
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Reset env variables
        self.step_ctr = 0
        self.episodes = 0

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[7:] = self.scale_action([0] * 18)
        init_q[0] = 0.2 # np.random.rand() * 4 - 4
        init_q[1] = 0.0 # np.random.rand() * 8 - 4
        init_q[2] = 0.3 + int(self.env_list == ["stairs_down"]) * 1.2
        init_qvel = np.zeros(self.qvel_dim)

        if init_pos is not None:
            init_q[0:3] += init_pos

        self.vel_sum = 0

        # Init_quat
        if self.rnd_init_yaw:
            self.rnd_yaw = np.random.rand() * 1. - 0.5
        else:
            self.rnd_yaw = 0

        rnd_quat = my_utils.rpy_to_quat(0,0,self.rnd_yaw)
        init_q[3:7] = rnd_quat

        self.prev_deviation = np.min((abs((self.rnd_yaw % 6.183) - (0 % 6.183)), abs(self.rnd_yaw - 0)))
        self.prev_y_deviation = 0

        # Set environment state
        self.set_state(init_q, init_qvel)

        ctr = 0
        while True:
            ctr += 1
            self.sim.forward()
            self.sim.step()
            #self.render()
            #time.sleep(0.01)

            if np.max(self.sim.get_state().qvel.tolist()[0:7]) < 0.15 and ctr > 10:
                break

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
        envs = np.random.choice(self.env_list, n_envs, replace=self.replace_envs)

        if n_envs == 1:
            size_list = [steps]
            scaled_indeces_list = [0]
        else:
            size_list = []
            raw_indeces = np.linspace(0, 1, n_envs + 1)[1:-1]
            current_idx = 0
            scaled_indeces_list = []
            for idx in raw_indeces:
                idx_scaled = int(steps * idx) + np.random.randint(0, int(steps/6)) - int(steps/12)
                scaled_indeces_list.append(idx_scaled)
                size_list.append(idx_scaled - current_idx)
                current_idx = idx_scaled
            size_list.append(steps - sum(size_list))

        maplist = []
        current_height = 0
        for m, s in zip(envs, size_list):
            hm, current_height = self.generate_heightmap(m, s, current_height)
            maplist.append(hm)
        total_hm = np.concatenate(maplist, 1)
        heighest_point = np.max(total_hm)
        height_SF = max(heighest_point / 255., 1)
        total_hm /= height_SF
        total_hm = np.clip(total_hm, 0, 255).astype(np.uint8)

        #Smoothen transitions
        bnd = 2
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

        with open(Hexapod.MODELPATH + "limited.xml", "r") as in_file:
            buf = in_file.readlines()

        with open(Hexapod.MODELPATH + "{}.xml".format(self.ID), "w") as out_file:
            for line in buf:
                if line.startswith('    <hfield name="hill"'):
                    out_file.write('    <hfield name="hill" file="{}.png" size="{} 1.2 {} 0.1" /> \n '.format(self.ID, self.env_scaling * self.n_envs, 0.6 * height_SF))
                elif line.startswith('    <geom name="floor" conaffinity="1" condim="3"'):
                    out_file.write('    <geom name="floor" conaffinity="1" condim="3" material="MatPlane" pos="{} 0 -.0" rgba="0.8 0.9 0.8 1" type="hfield" hfield="hill"/>'.format(self.env_scaling * self.n_envs * 0.7))
                else:
                    out_file.write(line)

        return envs, size_list, scaled_indeces_list


    def generate_heightmap(self, env_name, env_length, current_height):
        if env_name == "flat":
            hm = np.ones((self.env_width, env_length)) * current_height

        if env_name == "tiles":
            sf = 3
            hm = np.random.randint(0, 50,
                                   size=(self.env_width // sf, env_length // sf)).repeat(sf, axis=0).repeat(sf, axis=1)
            hm_pad = np.zeros((self.env_width, env_length))
            hm_pad[:hm.shape[0], :hm.shape[1]] = hm
            hm = hm_pad + current_height

        if env_name == "pipe":
            pipe_form = np.square(np.linspace(-1.2, 1.2, self.env_width))
            pipe_form = np.clip(pipe_form, 0, 1)
            hm = 255 * np.ones((self.env_width, env_length)) * pipe_form[np.newaxis, :].T
            hm += current_height

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
            hm = np.ones((self.env_width, env_length)) * current_height
            stair_height = 45
            stair_width = 4

            initial_offset = 0
            n_steps = math.floor(env_length / stair_width) - 1

            for i in range(n_steps):
                hm[:, initial_offset + i * stair_width: initial_offset  + i * stair_width + stair_width] = current_height
                current_height += stair_height

            hm[:, n_steps * stair_width:] = current_height

        if env_name == "stairs_down":
            stair_height = 45
            stair_width = 4
            initial_offset = 0
            n_steps = math.floor(env_length / stair_width) - 1
            current_height += stair_height * n_steps

            hm = np.ones((self.env_width, env_length)) * current_height

            for i in range(n_steps):
                hm[:, initial_offset + i * stair_width: initial_offset  + i * stair_width + stair_width] = current_height
                current_height -= stair_height

            hm[:, n_steps * stair_width:] = current_height

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
            obstacle_height = 50
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

            hm += current_height


        if env_name == "perlin":
            oSim = OpenSimplex(seed=int(time.time()))

            height = 100

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
            hm += current_height


        return hm, current_height


    def demo(self):

        scaler = 1.0

        import torch as T
        for i in range(1000):

            for i in range(30):
                obs = self.step(np.zeros((self.act_dim)), render=True)
            print(T.tensor(obs[0]).unsqueeze(0))
            for i in range(30):
                act = np.array([0., -scaler, scaler] * 6)
                obs = self.step(act, render=True)
            for i in range(30):
                obs = self.step(np.array([0., scaler, -scaler] * 6), render=True)
            print(T.tensor(obs[0]).unsqueeze(0))

            for i in range(30):
                obs = self.step(np.ones((self.act_dim)) * scaler, render=True)
            print(T.tensor(obs[0]).unsqueeze(0))
            for i in range(30):
                obs = self.step(np.ones((self.act_dim)) * -scaler, render=True)
            print(T.tensor(obs[0]).unsqueeze(0))
            print("Repeating...")


    def actuator_test(self):
        self.reset()

        print("Starting actuator test")


        for i in range(60):

            act = np.array([0, -1.4, 0.6] * 6)
            self.sim.data.ctrl[:] = act
            for i in range(200):
                self.sim.forward()
                self.sim.step()
                self.render()

            act[0] = 0.3
            self.sim.data.ctrl[:] = act
            for i in range(200):
                self.sim.forward()
                self.sim.step()
                self.render()

            act[0] = -0.3
            self.sim.data.ctrl[:] = act
            for i in range(200):
                self.sim.forward()
                self.sim.step()
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


    def test(self, policy, render=True, N=30, seed=None):

        # obs = np.array([1, 0, 0] * 3 + [0, 1, 0] * 3 + [0, 0, 0, 1])
        # action = policy(my_utils.to_tensor(obs, True)).detach()
        # exit()

        if seed is not None:
            self.setseed(seed)
        self.env_change_prob = 1
        rew = 0
        vel_rew = 0
        dist_rew = 0
        for i in range(N):
            obs = self.reset()
            cr = 0
            vr = 0
            dr = 0
            for j in range(int(self.max_steps)):
                #obs[0:18] = obs[0:18] + np.random.randn(18) * 0.3
                action = policy(my_utils.to_tensor(obs, True)).detach()
                obs, r, done, _ = self.step(action[0].numpy(), render=True)
                cr += r

            rew += cr
            vel_rew += vr
            dist_rew += dr
            if render:
                print("Total episode reward: {}".format(cr))
        if render:
            print("Total average reward = {}".format(rew / N))
        return rew / N, vel_rew / N, dist_rew / N


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


    def setseed(self, seed):
        np.random.seed(seed)


    def test_recurrent(self, policy, render=True, N=30, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.env_change_prob = 1

        rew = 0
        vel_rew = 0
        dist_rew = 0
        for i in range(N):
            obs = self.reset()
            h = None
            cr = 0
            vr = 0
            dr = 0
            for j in range(self.max_steps):
                action, h = policy((my_utils.to_tensor(obs, True).unsqueeze(0), h))
                obs, r, done, (r_v, r_d) = self.step(action[0].detach().numpy())
                cr += r
                vr += r_v
                dr = max(dr, r_d)

                time.sleep(0.000)
                if render:
                    self.render()

            rew += cr
            vel_rew += vr
            dist_rew += dr

            if render:
                print("Total episode reward: {}".format(cr))

        return rew / N, vel_rew / N, dist_rew / N


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


    def close(self):
        if self.viewer is not None:
            self.viewer = None


if __name__ == "__main__":
    hex = Hexapod()
    print(hex.obs_dim)
    print(hex.act_dim)
    hex.demo()