import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
from math import sqrt, acos, fabs
import random
import string


class Pendulum:
    def __init__(self, animate=False):
        self.modelpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/inverted_double_pendulum.xml")
        self.max_steps = 150
        self.mem_dim = 0
        self.cumulative_environment_reward = None

        self.model = mujoco_py.load_model_from_path(self.modelpath)
        self.sim = mujoco_py.MjSim(self.model)

        self.model.opt.timestep = 0.02

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = self.q_dim + self.qvel_dim
        self.act_dim = 1

        # Environent inner parameters
        self.viewer = None

        # Reset env variables
        self.step_ctr = 0
        self.episodes = 0

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


    def get_obs(self):
        qpos = self.sim.get_state().qpos.tolist()
        qvel = self.sim.get_state().qvel.tolist()
        return np.concatenate((np.asarray(qpos), np.clip(np.asarray(qvel), -10, 10)))

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
        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        obs = self.get_obs()
        r = np.abs(np.sin(obs[1]))

        done = self.step_ctr > self.max_steps

        return obs, r, done, None


    def reset(self):
        self.step_ctr = 0

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_qvel = np.zeros(self.qvel_dim, dtype=np.float32)

        if True:
            rnd_vec = np.random.rand() * 5.0 + 1.0
            self.model.body_mass[2] = rnd_vec
            self.model.geom_rgba[2] = [rnd_vec / 7.,0,0,1]

        # Set environment state
        self.set_state(init_q, init_qvel)
        obs = self.get_obs()

        return obs


    def demo(self):
        self.reset()
        for i in range(10000):
            obs, r, _, _ = self.step(np.random.randn(self.act_dim))
            #print(obs)
            #time.sleep(0.2)
            self.render()


    def test(self, policy):
        for i in range(100):
            obs = self.reset()
            cr = 0
            for j in range(self.max_steps):
                action = policy(my_utils.to_tensor(obs, True)).detach()
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
                action, h = policy((my_utils.to_tensor(obs, True).unsqueeze(0), h))
                obs, r, done, od, = self.step(action[0, 0].detach().numpy())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))


if __name__ == "__main__":
    cp = Pendulum(animate=True)
    print(cp.obs_dim)
    print(cp.act_dim)
    cp.demo()