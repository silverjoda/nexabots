import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
from math import sqrt, acos, fabs
from src.envs.hexapod_terrain_env.hf_gen import ManualGen, EvoGen, HMGen
import random
import string



class Hexapod:
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/hexapod_trossen_barrier.xml")
    def __init__(self, animate=False):

        print("Trossen hexapod")

        self.leg_list = ["coxa_fl_geom","coxa_fr_geom","coxa_rr_geom","coxa_rl_geom","coxa_mr_geom","coxa_ml_geom"]

        self.modelpath = Hexapod.MODELPATH
        self.max_steps = 600
        self.mem_dim = 18
        self.cumulative_environment_reward = None

        self.joints_rads_low = np.array([-0.6, -1., -1.] * 6)
        self.joints_rads_high = np.array([0.6, 0.3, 1.] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.model = mujoco_py.load_model_from_path(self.modelpath)
        self.sim = mujoco_py.MjSim(self.model)

        self.model.opt.timestep = 0.02

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = self.q_dim + self.qvel_dim - 2 + 6 + self.mem_dim - 18 - 6 - 1 - 1 + 3
        self.act_dim = self.sim.data.actuator_length.shape[0] + self.mem_dim

        # Environent inner parameters
        self.viewer = None

        # Reset env variables
        self.step_ctr = 0

        #self.envgen = ManualGen(12)
        #self.envgen = HMGen()
        #self.envgen = EvoGen(12)
        self.episodes = 0

        self.reset()

        # Initial methods
        if animate:
            self.setupcam()


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
        od['contacts'] = (np.abs(np.array(self.sim.data.cfrc_ext[[4, 7, 10, 13, 16, 19]])).sum(axis=1) > 0.05).astype(np.float32)
        #print(od['contacts'])
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

        # Mute appropriate leg joints
        for i in range(6):
            if self.dead_leg_vector[i] == 1:
                ctrl[i * 3:i * 3 + 3] = np.zeros(3) #np.random.randn(3) * 0.1

        if self.mem_dim == 0:
            mem = np.zeros(0)
            act = ctrl
            ctrl = self.scale_action(act)
        else:
            mem = ctrl[-self.mem_dim:]
            act = ctrl[:-self.mem_dim]
            ctrl = self.scale_action(act)

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        obs = self.get_obs()
        obs_dict = self.get_obs_dict()

        # Angle deviation
        x, y, z, qw, qx, qy, qz = obs[:7]

        xd, yd, zd, _, _, _ = self.sim.get_state().qvel.tolist()[:6]
        angle = 2 * acos(qw)

        # Reward conditions
        ctrl_effort = np.square(ctrl).sum()
        target_progress = xd
        target_vel = 0.3
        velocity_rew = 1. / (abs(xd - target_vel) + 1.) - 1. / (target_vel + 1.)
        height_pen = np.square(zd)

        contact_cost = 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        rV = (target_progress * 0.0,
              velocity_rew * 7.0,
              - ctrl_effort * 0.001,
              - np.square(angle) * 0.2,
              - np.square(yd) * 5.,
              - contact_cost * 0.0,
              - height_pen * 0.2 * int(self.step_ctr > 20))

        r = sum(rV)
        r = np.clip(r, -3, 3)
        obs_dict['rV'] = rV
        self.cumulative_environment_reward += r

        # Reevaluate termination condition
        done = self.step_ctr > self.max_steps# or (abs(angle) > 2.4 and self.step_ctr > 30) or abs(y) > 0.5 or x < -0.2

        obs = np.concatenate([np.array(self.sim.get_state().qpos.tolist()[3:]),
                              [xd, yd],
                              obs_dict["contacts"],
                              mem])

        if np.random.rand() < self.dead_leg_prob:
            idx = np.random.randint(0,6)
            self.dead_leg_vector[idx] = 1
            self.model.geom_rgba[self.model._geom_name2id[self.leg_list[idx]]] = [1, 0, 0, 1]
            self.dead_leg_prob = 0

        return obs, r, done, obs_dict


    def reset(self):

        self.cumulative_environment_reward = 0
        self.dead_leg_prob = 0.003
        self.dead_leg_vector = [0, 0, 0, 0, 0, 0]
        self.step_ctr = 0

        for i in range(6):
            self.model.geom_rgba[self.model._geom_name2id[self.leg_list[i]]] = [0.0, 0.6, 0.4, 1]

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = np.random.randn() * 0.1 + 0.05
        init_q[1] = np.random.randn() * 0.1
        init_q[2] = 0.15
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        # Set environment state
        self.set_state(init_q, init_qvel)

        obs_dict = self.get_obs_dict()
        obs = np.concatenate([np.array(self.sim.get_state().qpos.tolist()[3:]),
                              [0, 0],
                              obs_dict["contacts"],
                              np.zeros(self.mem_dim)])

        return obs


    def demo(self):
        self.reset()
        for i in range(1000):
            #self.step(np.random.randn(self.act_dim))
            for i in range(100):
                self.step(np.zeros((self.act_dim)))
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


    def test(self, policy):
        #self.envgen.load()
        for i in range(100):
            obs = self.reset()
            cr = 0
            for j in range(self.max_steps):
                action = policy(my_utils.to_tensor(obs, True)).detach()
                #print(action[0, :-self.mem_dim])
                obs, r, done, od, = self.step(action[0].numpy())
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
            for j in range(self.max_steps):
                action, h_ = policy((my_utils.to_tensor(obs, True), h))
                h = h_
                obs, r, done, od, = self.step(action[0].detach().numpy())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))

if __name__ == "__main__":
    ant = Hexapod(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()