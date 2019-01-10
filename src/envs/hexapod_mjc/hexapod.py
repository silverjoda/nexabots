import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
from math import sqrt, acos, fabs

class Hexapod:
    N = 8
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/hexapod.xml".format(N))
    def __init__(self, animate=False, sim=None):

        print([sqrt(l**2 + l**2) for l in [0.1, 0.2, 0.5]])

        self.N_links = 4

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

        self.obs_dim = self.q_dim + self.qvel_dim - 2
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Environent inner parameters
        self.viewer = None
        self.step_ctr = 0
        self.max_steps = 300

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
        a = qpos[2:] + qvel
        return np.asarray(a, dtype=np.float32)


    def get_obs_dict(self):
        od = {}
        # Intrinsic parameters
        for j in self.sim.model.joint_names:
            od[j + "_pos"] = self.sim.data.get_joint_qpos(j)
            od[j + "_vel"] = self.sim.data.get_joint_qvel(j)

        # # Contacts:
        # ctct_idces = []
        # for i in range(self.N_links * 2):
        #     ctct_idces.append(self.model._body_name2id["frontFoot_{}".format(i)])
        # od['contacts'] = np.clip(np.square(np.array(
        #     self.sim.data.cfrc_ext[ctct_idces])).sum(axis=1), 0, 1)

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

        obs_p = self.get_obs()

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        #print(self.sim.data.ncon) # Prints amount of current contacts
        obs_c = self.get_obs()

        # Reevaluate termination condition
        done = self.step_ctr > self.max_steps

        # Angle deviation
        qw, qx, qy, qz  = obs_c[1:5]
        angle = 2 * acos(qw)
        #print([qw, qx, qy, qz], angle)

        # Reward conditions
        ctrl_effort = np.square(ctrl).mean() * 0.03
        target_progress = (obs_p[0] - obs_c[0]) * 60

        r = target_progress - ctrl_effort - abs(angle)

        return obs_c.astype(np.float32), r, done, self.get_obs_dict()


    def demo(self):
        self.reset()
        for i in range(1000):
            self.step(np.random.randn(self.act_dim))
            #self.step(np.zeros((self.act_dim)))
            self.render()


    def test(self, policy):
        self.reset()
        for i in range(100):
            done = False
            obs, _ = self.reset()
            cr = 0
            while not done:
                action = policy(my_utils.to_tensor(obs, True)).detach()
                obs, r, done, od, = self.step(action[0])
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))


    def reset(self):

        # Reset env variables
        self.step_ctr = 0

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = np.random.randn() * 0.1
        init_q[1] = np.random.randn() * 0.1
        init_q[2] = 0.80 + np.random.rand() * 0.1
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        obs = np.concatenate((init_q[2:], init_qvel)).astype(np.float32)

        # Set environment state
        self.set_state(init_q, init_qvel)

        return obs, self.get_obs_dict()


if __name__ == "__main__":
    ant = Hexapod(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
