import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os

class QuadFeelersMjc:
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "quad_feelers.xml")
    def __init__(self, animate=False, sim=None):
        if sim is not None:
            self.sim = sim
            self.model = self.sim.model
        else:
            self.modelpath = QuadFeelersMjc.MODELPATH
            self.model = mujoco_py.load_model_from_path(self.modelpath)
            self.sim = mujoco_py.MjSim(self.model)

        self.model.opt.timestep = 0.04
        self.N_boxes = 4

        self.max_steps = 400

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = self.q_dim + self.qvel_dim - 7 * self.N_boxes - 6 * self.N_boxes + 7 - 2
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Environent inner parameters
        self.viewer = None
        self.step_ctr = 0

        self.joints_rads_low = np.array([-0.3, -1.] * 4 + [-1, -1, -1, -1])
        self.joints_rads_high = np.array([0.3, 0] * 4 + [1, 1, 1, 1])
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

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
        qpos = self.sim.get_state().qpos.tolist()[: - 7 * self.N_boxes]
        qvel = self.sim.get_state().qvel.tolist()[: - 6 * self.N_boxes]
        a = qpos + qvel
        return np.asarray(a, dtype=np.float32)


    def get_obs_dict(self):
        od = {}
        # Intrinsic parameters
        for j in self.sim.model.joint_names:
            od[j + "_pos"] = self.sim.data.get_joint_qpos(j)
            od[j + "_vel"] = self.sim.data.get_joint_qvel(j)

        # Contacts:
        od['contacts'] = np.clip(np.square(np.array(self.sim.data.cfrc_ext[[3, 5, 7, 9, 11, 13]])).sum(axis=1), 0, 1)
        #print(od['contacts'])
        od['torso_contact'] = np.clip(np.square(np.array(self.sim.data.cfrc_ext[1])).sum(axis=0), 0, 1)

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

        #print(self.sim.data.ncon) # Prints amount of current contacts
        obs_c = self.get_robot_obs()
        obs_dict = self.get_obs_dict()

        x, y, z, qw, qx, qy, qz = obs_c[:7]
        angle = 2 * np.arccos(qw)

        # Reevaluate termination condition.
        done = self.step_ctr > self.max_steps or obs_dict['torso_contact'] > 0.1 or ((obs_dict['contacts'] > 0).sum() < 2 and self.step_ctr > 15)

        xd, yd, _, _, _, _ = obs_dict["root_vel"]

        ctrl_effort = np.square(ctrl[0:8]).mean() * 0.00
        target_progress = xd * 1.
        survival = 0.3

        obs = np.concatenate((obs_c.astype(np.float32)[2:], obs_dict["contacts"], [obs_dict['torso_contact']]))
        r = target_progress - ctrl_effort + obs_dict["contacts"][-2:].sum() * 0.01 - obs_dict['torso_contact'] * 0.2 - angle * 0.0 + survival

        return obs, r, done, obs_dict


    def demo(self):
        self.reset()
        for i in range(1000):
            # self.step(np.random.randn(self.act_dim))
            for i in range(100):
                self.step(np.ones((self.act_dim)) * 0)
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
            cr = 0
            while not done:
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
            obs = self.reset()
            h = policy.init_hidden()
            cr = 0
            self.max_steps = 600
            import matplotlib.pyplot as plt
            #fig = plt.figure()
            acts = []
            while not done:
                action, h_ = policy((my_utils.to_tensor(obs, True), h))
                acts.append(action[0].detach())
                h = h_
                obs, r, done, od, = self.step(action[0].detach())
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

        r = self.q_dim - self.N_boxes * 7
        for i in range(self.N_boxes):
            init_q[r + i * 7 :r + i * 7 + 3] = [i * 2.2 + 1.7, np.clip(np.random.randn() * 1.0, -1.8, 1.8), 0.6]

        # Set environment state
        self.set_state(init_q, init_qvel)
        obs = np.concatenate((init_q[: - 7 * self.N_boxes], init_qvel[: - 6 * self.N_boxes])).astype(np.float32)

        obs_dict = self.get_obs_dict()
        obs = np.concatenate((obs[2:], obs_dict["contacts"], [obs_dict["torso_contact"]]))

        return obs


if __name__ == "__main__":
    ant = QuadFeelersMjc(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
