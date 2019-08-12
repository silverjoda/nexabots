import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os

class Centipede:
    def __init__(self, N_LINKS):

        self.N_links = N_LINKS
        MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/Centipede{}.xml".format(N_LINKS * 2))

        self.modelpath = MODELPATH
        self.model = mujoco_py.load_model_from_path(self.modelpath)
        self.sim = mujoco_py.MjSim(self.model)

        self.model.opt.timestep = 0.02

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        # j, jd, ctcts
        self.obs_dim = self.N_links * 6 - 2 + self.N_links * 6 - 2 + self.N_links * 2 + 10
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Environent inner parameters
        self.viewer = None
        self.step_ctr = 0
        self.max_steps = 200

        self.joints_rads_low = np.array([-0.65, 0.5, -0.65, 0.5] + [-0.2, -0.15, -0.65, 0.5, -0.65, 0.5] * (self.N_links - 1))
        self.joints_rads_high = np.array([0.65, 1.6, 0.65, 1.6] + [0.2, 0.3, 0.65, 1.6, 0.65, 1.6] * (self.N_links - 1))
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

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


    def scale_action(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low


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

        od['contacts'] = (np.abs(np.array(self.sim.data.cfrc_ext[ctct_idces])).sum(axis=1) > 0.05).astype(
            np.float32)
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
            self._setupcam()
        self.viewer.render()


    def step(self, ctrl):
        ctrl = self.scale_action(ctrl)

        # This is for additive control
        #joint_list = np.array(self.sim.get_state().qpos.tolist()[7:31])
        #joint_list += ctrl

        ctrl = np.clip(ctrl, self.joints_rads_low, self.joints_rads_high)

        self.sim.data.ctrl[:] = ctrl
        self.sim.step()
        self.step_ctr += 1

        vel = self.sim.get_state().qvel.tolist()
        pos = self.sim.get_state().qpos.tolist()

        xd, yd = vel[:2]
        #x, y, z, q0, q1, q2, q3 = pos[:7]
        #roll, pitch, yaw = my_utils.quat_to_rpy((q0,q1,q2,q3))

        # Reevaluate termination condition
        done = self.step_ctr >= self.max_steps #or yaw > 0.9 or z > 1.2 or z < 0.2 or abs(y) > 0.8

        ctrl_effort = np.square(ctrl).mean() * 0.5
        #print(ctrl_effort)
        #target_progress = -vel[0]
        target_vel = 1.5
        velocity_rew = 1. / (abs(-xd - target_vel) + 1.) - 1. / (target_vel + 1.)

        obs_dict = self.get_obs_dict()
        obs = np.concatenate((pos[7:], vel[6:], obs_dict["contacts"], pos[3:7], vel[:6]))

        r = velocity_rew - ctrl_effort - abs(yd) * 0.05

        return obs, r, done, self.get_obs_dict()


    def reset(self):

        # Reset env variables
        self.step_ctr = 0

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = 0
        init_q[1] = 0
        init_q[2] = 0.80
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        # Set environment state
        self._set_state(init_q, init_qvel)

        for i in range(10):
            self.sim.step()

        obs, _, _, _ = self.step([0] * self.act_dim)

        return obs


    def demo(self):
        self.reset()
        for i in range(1000):
            #self.step(np.random.randn(self.act_dim))
            #self.render()
            for i in range(200):
                self.step(np.ones(self.act_dim) * 0)
                self.render()
            for i in range(200):
                self.step(np.ones(self.act_dim) * 1)
                self.render()
            for i in range(200):
                self.step(np.ones(self.act_dim) * -1)
                self.render()


    def test(self, policy):
        self.reset()
        for i in range(100):
            done = False
            obs = self.reset()
            cr = 0
            for i in range(1000):
                action = policy(my_utils.to_tensor(obs, True))[0].detach().numpy()
                obs, r, done, od, = self.step(action)
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))



if __name__ == "__main__":
    cp = Centipede(4)
    print(cp.obs_dim)
    print(cp.act_dim)
    cp.demo()
