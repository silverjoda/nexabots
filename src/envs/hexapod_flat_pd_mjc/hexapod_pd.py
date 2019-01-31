import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
from math import sqrt, acos, fabs

class Hexapod:
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/hexapod.xml")
    def __init__(self, animate=False, sim=None):

        print([sqrt(l**2 + l**2) for l in [0.1, 0.2, 0.5]])

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

        # Environent inner parameters
        self.viewer = None
        self.step_ctr = 0
        self.max_steps = 400
        self.ctrl_vecs = []
        self.dead_joint_idx = 0
        self.dead_leg_idx = 0

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

        #ctrl[self.dead_joint_idx] = 0
        #if np.random.rand() < 0.3:
        #    ctrl[self.dead_leg_idx * 3: (self.dead_leg_idx + 1) * 3] = 0

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

        xd, yd, _, _, _, _ = obs_dict["root_vel"]
        angle = 2 * acos(qw)

        # Reward conditions
        ctrl_effort = np.square(ctrl).sum()
        target_progress = xd
        velocity_pen = np.square(xd - 1)
        height_pen = np.square(z - 0.5)

        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        rV = (target_progress * 1,
              - ctrl_effort * 0.03,
              - np.square(angle) * 1.5,
              - abs(yd) * 0.1,
              - contact_cost * 0,
              - velocity_pen * 0,
              height_pen * 2.5)

        r = sum(rV)

        obs_dict['rV'] = rV

        # Reevaluate termination condition
        done = self.step_ctr > self.max_steps or abs(angle) > 0.7 or abs(y) > 0.5

        # if done:
        #     ctrl_sum = np.zeros(self.act_dim)
        #     for cv in self.ctrl_vecs:
        #         ctrl_sum += np.abs(np.array(cv))
        #     ctrl_dev = np.abs(ctrl_sum - ctrl_sum.mean()).mean()
        #
        #     r -= ctrl_dev * 3

        obs = np.concatenate((obs.astype(np.float32)[2:], obs_dict["contacts"]))

        return obs, r, done, obs_dict


    def demo(self):
        self.reset()
        for i in range(1000):
            #self.step(np.random.randn(self.act_dim))
            self.step(-np.ones((self.act_dim)))
            print(np.around(self.sim.get_state().qpos.tolist(),2))
            self.render()


    def test(self, policy):
        self.reset()
        for i in range(100):
            done = False
            obs = self.reset()
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
            done = False
            obs, _ = self.reset()
            h = policy.init_hidden()
            cr = 0
            for j in range(self.max_steps):
                action, h_ = policy((my_utils.to_tensor(obs, True), h))
                h = h_
                obs, r, done, od, = self.step(action[0].detach())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))


    def reset(self): #
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


if __name__ == "__main__":
    ant = Hexapod(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
