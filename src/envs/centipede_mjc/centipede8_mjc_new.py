import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os

class CentipedeMjc8:
    N = 8
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/Centipede{}.xml".format(N))
    def __init__(self, animate=False, sim=None):

        self.N_links = 4

        #
        if sim is not None:
            self.sim = sim
            self.model = self.sim.model
        else:
            self.modelpath = CentipedeMjc8.MODELPATH
            self.model = mujoco_py.load_model_from_path(self.modelpath)
            self.sim = mujoco_py.MjSim(self.model)

        self.model.opt.timestep = 0.02

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        # q -2 + 5 + q_ -2 + 6 + contacts
        self.obs_dim = self.N_links * 6 - 2 + 5 + self.N_links * 6 - 2 + 6 + self.N_links * 2
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Environent inner parameters
        self.viewer = None
        self.step_ctr = 0
        self.max_steps = 200

        # Initial methods
        if animate:
            self._setupcam()

        self.reset()


    def _setupcam(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 1.3
        self.viewer.cam.lookat[0] = -0.1
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -20


    def _get_jointvals(self):
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

        # Contacts:
        ctct_idces = []
        for i in range(self.N_links * 2):
            ctct_idces.append(self.model._body_name2id["frontFoot_{}".format(i)])
        od['contacts'] = np.clip(np.square(np.array(
            self.sim.data.cfrc_ext[ctct_idces])).sum(axis=1), 0, 1)

        return od


    def _set_state(self, qpos, qvel=None):
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

        torso_p = self.sim.get_state().qpos.tolist()

        self.sim.data.ctrl[:] = ctrl
        self.sim.step()
        self.step_ctr += 1

        torso_c = self.sim.get_state().qpos.tolist()

        # Reevaluate termination condition
        done = self.step_ctr >= self.max_steps

        ctrl_effort = np.square(ctrl).sum() * 0.1
        joint_velocity_pen = np.square(self.sim.get_state().qvel[6:]).sum() * 0.001
        #print(joint_velocity_pen)
        target_progress = (torso_p[0] - torso_c[0]) * 60

        obs_dict = self.get_obs_dict()
        obs = np.concatenate((self._get_jointvals().astype(np.float32), obs_dict["contacts"]))

        r = target_progress - ctrl_effort - joint_velocity_pen


        return obs, r, done, self.get_obs_dict()


    def demo(self):
        self.reset()
        for i in range(1000):
            self.step(np.random.randn(self.act_dim))
            self.render()


    def test(self, policy):
        self.reset()
        for i in range(100):
            done = False
            obs, _ = self.reset()
            cr = 0
            for i in range(1000):
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
            while not done:
                action, h_ = policy((my_utils.to_tensor(obs, True), h))
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

        obs = np.concatenate((init_q[2:], init_qvel)).astype(np.float32)

        obs_dict = self.get_obs_dict()
        obs = np.concatenate((obs, obs_dict["contacts"]))

        # Set environment state
        self._set_state(init_q, init_qvel)

        return obs, obs_dict


if __name__ == "__main__":
    ant = CentipedeMjc8(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
