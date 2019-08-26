import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
import src.envs.ant_feelers_mem_mjc.xml_gen as xml_gen


class AntFeelersMjc:
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "afgm_env.xml")
    def __init__(self, animate=False):
        self.animate = animate
        self.modelpath = AntFeelersMjc.MODELPATH
        self.xmlgen = xml_gen.Gen()

        # Environent inner parameters
        self.step_ctr = 0
        self.N_boxes = 5
        self.max_steps = 600
        self.mem_dim = 0

        self.joints_rads_low = np.array([-0.7, 0.8] * 4 + [-1, -1, -1, -1])
        self.joints_rads_high = np.array([0.7, 1.4] * 4 + [1, 1, 1, 1])
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

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
        od['contacts'] = np.clip(np.square(np.array(self.sim.data.cfrc_ext[[4, 7, 10, 13, 15, 17]])).sum(axis=1), 0, 1)
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
        goal_dist_pre = np.abs(self.goal_pos - self.sim.get_state().qpos.tolist()[:2]).sum()

        mem = ctrl[-self.mem_dim:]
        act = ctrl[:-self.mem_dim]
        sact = self.scale_action(act)

        self.sim.data.ctrl[:] = sact
        self.sim.step()
        self.step_ctr += 1

        goal_dist_post = np.abs(self.goal_pos - self.sim.get_state().qpos.tolist()[:2]).sum()

        obs_c = self.get_obs()
        obs_dict = self.get_obs_dict()

        x, y, z, qw, qx, qy, qz = obs_c[:7]
        #angle = 2 * np.arccos(qw)
        _, _, yaw = my_utils.quat_to_rpy((qw, qx, qy, qz))

        target_angle = np.arctan2(self.goal_pos[1] - y, self.goal_pos[0] - x)
        target_err = abs(yaw - target_angle)

        # Reevaluate termination condition.
        done = self.step_ctr > self.max_steps # or obs_dict['torso_contact'] > 0.1

        xd, yd, _, _, _, _ = obs_dict["root_vel"]

        ctrl_effort = np.square(ctrl[0:8]).mean() * 0.00
        target_progress = (goal_dist_pre - goal_dist_post) * 50
        #print(target_progress)

        obs = np.concatenate((self.goal_pos - obs_c[:2], obs_c[2:], obs_dict["contacts"], [obs_dict['torso_contact']], mem))
        r = np.clip(target_progress,-1, 1) - ctrl_effort - min(target_err, 3) * 0.1

        return obs, r, done, obs_dict


    def reset(self):
        # Goal pos
        self.goal_pos = (np.random.rand(2) * 4 + 2) * (np.sign(np.random.randn(2)))

        # Generate new environment
        self.xmlgen.generate(self.N_boxes, self.goal_pos)

        # Reset the model
        while True:
            try:
                self.model = mujoco_py.load_model_from_path(self.modelpath)
                break
            except Exception:
                pass

        self.sim = mujoco_py.MjSim(self.model)
        self.model.opt.timestep = 0.04

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]


        self.obs_dim = self.q_dim + self.qvel_dim + 7 + self.mem_dim
        self.act_dim = self.sim.data.actuator_length.shape[0] + self.mem_dim

        # Reset env variables
        self.step_ctr = 0
        self.viewer = None

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = np.random.randn() * 0.1
        init_q[1] = np.random.randn() * 0.1
        init_q[2] = 0.80 + np.random.rand() * 0.1
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        # Set environment state
        self.set_state(init_q, init_qvel)
        obs = np.concatenate((init_q, init_qvel)).astype(np.float32)

        obs_dict = self.get_obs_dict()
        obs = np.concatenate((self.goal_pos - obs[:2], obs[2:], obs_dict["contacts"], [obs_dict["torso_contact"]], np.zeros(self.mem_dim)))


        return obs


    def demo(self):
        self.reset()
        for i in range(100):
            act = np.ones(self.act_dim)
            self.step(act)
            self.render()

        for i in range(100):
            act = np.ones(self.act_dim) * -1
            self.step(act)
            self.render()

        for i in range(100):
            act = np.zeros(self.act_dim)
            self.step(act)
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
        N = 20
        rew = 0
        for i in range(N):
            h_list = []
            obs = self.reset()
            h = None
            cr = 0
            for j in range(self.max_steps):
                action, h = policy((my_utils.to_tensor(obs, True).unsqueeze(0), h))
                obs, r, done, od, = self.step(action[0].detach().numpy())
                cr += r
                rew += r
                time.sleep(0.001)
                self.render()
                # h_list.append(h[0][:,0,:].detach().numpy())
            print("Total episode reward: {}".format(cr))
            # h_arr = np.stack(h_list)
            # h_episodes.append(h_arr)

        print("Total average reward = {}".format(rew / N))

if __name__ == "__main__":
    ant = AntFeelersMjc(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
