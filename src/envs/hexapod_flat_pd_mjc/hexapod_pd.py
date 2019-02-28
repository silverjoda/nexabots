import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
from math import sqrt, acos, fabs
import queue

import socket

if socket.gethostname() != "goedel":
    import gym
    from gym import spaces
    from gym.utils import seeding

class Hexapod():
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/hexapod.xml")
    def __init__(self, animate=False, sim=None):

        print([sqrt(l**2 + l**2) for l in [0.1, 0.3, 0.4]])

        if sim is not None:
            self.sim = sim
            self.model = self.sim.model
        else:
            self.modelpath = Hexapod.MODELPATH
            self.model = mujoco_py.load_model_from_path(self.modelpath)
            self.sim = mujoco_py.MjSim(self.model)

        self.model.opt.timestep = 0.02

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = self.q_dim + self.qvel_dim - 2 + 6
        self.act_dim = self.sim.data.actuator_length.shape[0]

        if socket.gethostname() != "goedel":
            self.render_prob = 0.00
            self.observation_space = spaces.Box(low=-10, high=10, dtype=np.float32, shape=(self.obs_dim,))
            self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.act_dim,))

        # Environent inner parameters
        self.viewer = None
        self.step_ctr = 0
        self.max_steps = 300

        self.ctrl_vecs = []

        self.joints_rads_low = np.array([-0.3, -1., 1.] * 6)
        self.joints_rads_high = np.array([0.3, 0, 2.] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.rew_len = 1200
        self.rew_list = [0] * self.rew_len
        self.n_episodes = 0

        # Initial methods
        if animate:
            self.setupcam()

        self.reset()


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
        od['contacts'] = np.clip(np.square(np.array(self.sim.data.cfrc_ext[[4, 7, 10, 13, 16, 19]])).sum(axis=1), 0, 1)
        #od['contacts'] = np.zeros(6)
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
        ctrl = self.scale_action(ctrl)

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        self.ctrl_vecs.append(ctrl)

        #print(self.sim.data.ncon) # Prints amount of current contacts

        obs = self.get_obs()
        obs_dict = self.get_obs_dict()

        # Angle deviation
        x, y, z, qw, qx, qy, qz = obs[:7]

        xd, yd, zd, _, _, _ = self.sim.get_state().qvel.tolist()[:6]
        angle = 2 * acos(qw)

        # Reward conditions
        ctrl_effort = np.square(ctrl).sum()
        target_progress = xd
        target_vel = 1.0
        velocity_rew = 1. / (abs(xd - target_vel) + 1.) - 1. / (target_vel + 1.)
        height_pen = np.square(zd)

        contact_cost = 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        rV = (target_progress * 0.0,
              velocity_rew * 3.0,
              - ctrl_effort * 0.01,
              - np.square(angle) * 0.1,
              - abs(yd) * 0.01,
              - contact_cost * 0.0,
              - height_pen * 0.1)

        # 1.0 with 0.1 pens # 1.4 with 0.3 pens # 1.2 with 0.9 pens
        #print(rV)

        r = sum(rV)
        obs_dict['rV'] = rV

        # Reevaluate termination condition
        done = self.step_ctr > self.max_steps or (abs(angle) > 0.9 and self.step_ctr > 30) or abs(y) > 0.7
        obs = np.concatenate((obs.astype(np.float32)[2:], obs_dict["contacts"]))
        #
        # self.rew_list[self.n_episodes % self.rew_len] = r
        # rew_mean = sum(self.rew_list) / self.rew_len
        # self.n_episodes += 1
        #
        # print(rew_mean)

        return obs, r, done, obs_dict


    def reset(self):  #
        # Reset env variables
        self.step_ctr = 0
        self.ctrl_vecs = []
        self.dead_joint_idx = np.random.randint(0, self.act_dim)
        self.dead_leg_idx = np.random.randint(0, self.act_dim / 3)

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = np.random.randn() * 0.1
        init_q[1] = np.random.randn() * 0.1
        init_q[2] = 0.80 + np.random.rand() * 0.1
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        obs = np.concatenate((init_q[2:], init_qvel)).astype(np.float32)

        # Set environment state
        self.set_state(init_q, init_qvel)

        obs_dict = self.get_obs_dict()
        obs = np.concatenate((obs, obs_dict["contacts"]))

        return obs

    def demo(self):
        self.reset()
        for i in range(1000):
            #self.step(np.random.randn(self.act_dim))
            for i in range(100):
                self.step(np.ones((self.act_dim)) * 0)
                self.render()
            for i in range(100):
                self.step(np.array([0, -1, 1] * 6))
                self.render()
            for i in range(100):
                self.step(np.ones((self.act_dim)) * 1)
                self.render()
            for i in range(100):
                self.step(np.ones((self.act_dim)) * -1)
                self.render()


    def test(self, policy):
        self.reset()
        for i in range(100):
            done = False
            obs = self.reset()
            self.max_steps = 800
            cr = 0
            for j in range(self.max_steps):
                action = policy(my_utils.to_tensor(obs, True)).detach()
                obs, r, done, od, = self.step(action[0])
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))


    def test_recurrent(self, policy):
        self.reset()
        for i in range(100):
            obs = self.reset()
            h = None
            cr = 0
            for j in range(self.max_steps * 2):
                action, h_ = policy((my_utils.to_tensor(obs, True), h))
                h = h_
                obs, r, done, od, = self.step(action[0].detach())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))




if __name__ == "__main__":
    ant = Hexapod(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
