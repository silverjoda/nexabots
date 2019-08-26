import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
import cv2
from src.envs.locom_benchmarks import hf_gen

class Hexapod():
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "assets/hex_test.xml")

    def __init__(self):
        print("Locomotion terrain tester")

        # Generate environment
        res = 1
        hm = hf_gen.hm_corridor_various_width(res)
        cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "assets/hm_test.png"), hm)

        # External parameters
        self.joints_rads_low = np.array([-0.6, -1.4, -0.8] * 6)
        self.joints_rads_high = np.array([0.6, 0.4, 0.8] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        # Load simulator
        self.model = mujoco_py.load_model_from_path(Hexapod.MODELPATH)
        self.sim = mujoco_py.MjSim(self.model)
        self.model.opt.timestep = 0.02
        self.viewer = None

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        # Set initial position
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = 0.0
        init_q[1] = 0.0
        init_q[2] = 0.50
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        # Set environment state
        self.set_state(init_q, init_qvel)


    def setupcam(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * .3
        self.viewer.cam.lookat[0] = -0.1
        self.viewer.cam.lookat[1] = -1
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -30


    def get_state(self):
        return self.sim.get_state()


    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.q_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()


    def step(self):
        self.sim.forward()
        self.sim.step()


    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()


    def demo(self):
        for i in range(100000):
            self.step()
            self.render()


if __name__ == "__main__":
    hex = Hexapod()
    hex.demo()