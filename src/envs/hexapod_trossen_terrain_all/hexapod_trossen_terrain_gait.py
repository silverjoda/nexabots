import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
import cv2
from math import sqrt, acos, fabs
from src.envs.hexapod_terrain_env.hf_gen import ManualGen, EvoGen, HMGen
import random
import string

# import gym
# from gym import spaces
# from gym.utils import seeding

class Hexapod():
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "assets/hexapod_trossen_")

    def __init__(self, env_list=None, max_n_envs=3):
        print("Trossen hexapod envs: {}".format(env_list))

        self.modelpath = Hexapod.MODELPATH
        self.max_steps = 250

        self.episode_reward = 0
        self.max_episode_reward = 0
        self.episodes = 0

        self.joints_rads_low = np.array([-0.3, -1.0, -1.0] * 6)
        self.joints_rads_high = np.array([0.3, 0.2, 0.4] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.viewer = None

        path = Hexapod.MODELPATH + "gait.xml"

        self.model = mujoco_py.load_model_from_path(path)
        self.sim = mujoco_py.MjSim(self.model)

        self.model.opt.timestep = 0.02

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = 18 * 2 + 6 + 4 + 6
        self.act_dim = self.sim.data.actuator_length.shape[0]

        self.reset()

        #self.observation_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.obs_dim,))
        #self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.act_dim,))


    def setupcam(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 1.3
        self.viewer.cam.lookat[0] = -0.1
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -20


    def scale_action(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low


    def scale_joints(self, joints):
        sjoints = np.array(joints)
        sjoints = ((sjoints - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1
        return sjoints


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
        ctrl_penalty = np.mean(ctrl)
        ctrl = self.scale_action(ctrl)

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        obs = self.get_obs()

        self.used_energy += self.sim.data.actuator_force

        # Angle deviation
        x, y, z, qw, qx, qy, qz = obs[:7]
        xd, yd, zd, thd, phid, psid = self.sim.get_state().qvel.tolist()[:6]

        # Reward conditions
        target_vel = 0.30
        velocity_rew_x = 1. / (abs(xd - target_vel) + 1.) - 1. / (target_vel + 1.)

        roll, pitch, yaw = my_utils.quat_to_rpy([qw,qx,qy,qz])

        contacts = (np.abs(np.array(self.sim.data.cfrc_ext[[4, 7, 10, 13, 16, 19]])).sum(axis=1) > 0.05).astype(
            np.float32)

        # mean_used_energy = self.used_energy / self.step_ctr
        # mean_used_energy_coxa = mean_used_energy[0::3]
        # mean_used_energy_femur = mean_used_energy[1::3]
        # mean_used_energy_tibia = mean_used_energy[2::3]
        # coxa_penalty = np.mean(np.square(mean_used_energy_coxa - np.mean(mean_used_energy_coxa)))
        # femur_penalty = np.mean(np.square(mean_used_energy_femur - np.mean(mean_used_energy_femur)))
        # tibia_penalty = np.mean(np.square(mean_used_energy_tibia - np.mean(mean_used_energy_tibia)))

        r_pos = velocity_rew_x * 5. # + np.mean(contacts) * 0.2
        r_neg = np.square(roll) * 1. + \
                np.square(pitch) * 1. + \
                np.square(zd) * 1. + \
                np.square(yd) * 4. + \
                np.square(y) * 7. + \
                np.square(yaw) * 4. + \
                np.square(self.sim.data.actuator_force).mean() * 0.000 + \
                ctrl_penalty * 0.0 # + \
                #(coxa_penalty * .1 + femur_penalty * .1 + tibia_penalty * .1) * self.step_ctr > 30


        r_neg = np.clip(r_neg, 0, 1) * 1.
        r_pos = np.clip(r_pos, -2, 2)
        r = r_pos - r_neg
        self.episode_reward += r

        # Reevaluate termination condition
        done = self.step_ctr > self.max_steps

        obs = np.concatenate([self.scale_joints(self.sim.get_state().qpos.tolist()[7:]),
                              self.sim.get_state().qvel.tolist()[6:],
                              self.sim.get_state().qvel.tolist()[:6],
                              [roll, pitch, yaw, y],
                              contacts])

        return obs, r, done, None


    def reset(self, init_pos = None):
        # Reset env variables
        self.step_ctr = 0
        self.episodes += 1
        self.used_energy = np.zeros((self.act_dim))

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = 0.0 # np.random.rand() * 4 - 4
        init_q[1] = 0.0 # np.random.rand() * 8 - 4
        init_q[2] = 0.12
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        # Init_quat
        self.rnd_yaw = np.random.rand() * 0.0 - 0.0
        rnd_quat = my_utils.rpy_to_quat(0, 0, self.rnd_yaw)
        init_q[3:7] = rnd_quat

        # Set environment state
        self.set_state(init_q, init_qvel)

        for i in range(10):
            self.sim.forward()
            self.sim.step()

        # self.render()
        # time.sleep(3)

        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs


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


    def test(self, policy):
        #self.envgen.load()
        self.env_change_prob = 1
        for i in range(100):
            obs = self.reset()
            cr = 0
            for j in range(int(self.max_steps * 1.5)):
                action = policy(my_utils.to_tensor(obs, True)).detach()
                obs, r, done, od, = self.step(action[0].numpy())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))


    def test_recurrent(self, policy):
        self.env_change_prob = 1
        self.reset()
        h_episodes = []
        for i in range(10):
            h_list = []
            obs = self.reset()
            h = None
            cr = 0
            for j in range(self.max_steps * 2):
                action, h = policy((my_utils.to_tensor(obs, True).unsqueeze(0), h))
                obs, r, done, od, = self.step(action[0,0].detach().numpy() + np.random.randn(self.act_dim) * 0.1)
                cr += r
                time.sleep(0.001)
                self.render()
                h_list.append(h[0][:,0,:].detach().numpy())
            print("Total episode reward: {}".format(cr))
            h_arr = np.stack(h_list)
            h_episodes.append(h_arr)

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