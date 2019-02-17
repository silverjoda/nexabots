import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os

class TestEnv:
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_env.xml")
    def __init__(self, animate=False, sim=None):
        if sim is not None:
            self.sim = sim
            self.model = self.sim.model
        else:
            self.modelpath = TestEnv.MODELPATH
            self.model = mujoco_py.load_model_from_path(self.modelpath)
            self.sim = mujoco_py.MjSim(self.model)

        self.model.opt.timestep = 0.02
        self.max_steps = 400

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = self.q_dim + self.qvel_dim
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Environent inner parameters
        self.viewer = None
        self.step_ctr = 0

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


    def get_robot_obs(self):
        qpos = self.sim.get_state().qpos.tolist()[:]
        qvel = self.sim.get_state().qvel.tolist()[:]
        a = qpos + qvel
        return np.asarray(a, dtype=np.float32)


    def get_obs_dict(self):
        od = {}
        # Intrinsic parameters
        for j in self.sim.model.joint_names:
            od[j + "_pos"] = self.sim.data.get_joint_qpos(j)
            od[j + "_vel"] = self.sim.data.get_joint_qvel(j)
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
        self.sim.data.ctrl[:] = ctrl
        self.sim.step()
        self.step_ctr += 1

        return self.get_robot_obs, 0, False, self.get_obs_dict()


    def demo(self):
        while True:
            print("A")
            for i in range(100):
                act = 1
                self.step(act)
                self.render()

            print("B")
            for i in range(100):
                act = 0
                self.step(act)
                self.render()

            print("C")
            for i in range(100):
                act = -1
                self.step(act)
                self.render()





    def reset(self):
        return


if __name__ == "__main__":
    ant = TestEnv(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
