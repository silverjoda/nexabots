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
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/hexapod_trossen.xml")
    def __init__(self, animate=False):

        print("Trossen hexapod")

        self.leg_list = ["coxa_fl_geom","coxa_fr_geom","coxa_rr_geom","coxa_rl_geom","coxa_mr_geom","coxa_ml_geom"]

        self.modelpath = Hexapod.MODELPATH
        self.max_steps = 400
        self.mem_dim = 0
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

        self.obs_dim = self.q_dim + self.qvel_dim - 2 + 6 + self.mem_dim + 2
        self.act_dim = self.sim.data.actuator_length.shape[0] + self.mem_dim

        # Environent inner parameters
        self.viewer = None

        # Reset env variables
        self.step_ctr = 0
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
        if self.mem_dim == 0:
            mem = np.zeros(0)
            act = ctrl
            ctrl = self.scale_action(act)
        else:
            mem = ctrl[-self.mem_dim:]
            act = ctrl[:-self.mem_dim]
            ctrl = self.scale_action(act)

        x, y, z, q0, q1, q2, q3 = self.get_obs()[:7]
        prev_dist_to_goal = np.sqrt(np.square(self.goal_xy[0] - x) + np.square(self.goal_xy[1] - y))

        siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
        yaw_prev = np.arctan2(siny_cosp, cosy_cosp)

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        obs = self.get_obs()
        obs_dict = self.get_obs_dict()

        x, y, z, q0, q1, q2, q3 = obs[:7]
        curr_dist_to_goal = np.sqrt(np.square(self.goal_xy[0] - x) + np.square(self.goal_xy[1] - y))

        xd, yd, zd, _, _, _ = self.sim.get_state().qvel.tolist()[:6]

        #roll, pitch, yaw = my_utils.quat_to_rpy((q0,q1,q2,q3))
        #print(roll, pitch, yaw)

        siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Calculate target angle to goal
        tar_angle = np.arctan2(self.goal_xy[1] - y, self.goal_xy[0] - x)

        # Reward conditions
        goal_velocity = (prev_dist_to_goal - curr_dist_to_goal) * 50

        target_velocity = 0.3
        velocity_rew = 1. / (abs(goal_velocity - target_velocity) + 1.) - 1. / (target_velocity + 1.)
        direction_rew = abs(tar_angle - yaw_prev) - abs(tar_angle - yaw)
        direction_pen = abs(tar_angle - yaw)

        rV = (velocity_rew * 3.0,
              goal_velocity * 0,
              direction_rew * 0,
              - direction_pen * .0,
              - np.square(ctrl).sum() * 0.00)
              #,- (np.square(pitch) * 1. + np.square(roll) * 1. + np.square(zd) * 1.) * self.goal_level * int(self.step_ctr > 15))

        r = sum(rV)
        r = np.clip(r, -3, 3)
        obs_dict['rV'] = rV

        # Reevaluate termination condition
        done = self.step_ctr > self.max_steps # or (abs(angle) > 2.4 and self.step_ctr > 30) or abs(y) > 0.5 or x < -0.2

        obs = np.concatenate([self.sim.get_state().qpos.tolist()[2:],
                              self.sim.get_state().qvel.tolist(),
                              obs_dict["contacts"],
                              [self.goal_xy[0] - x, self.goal_xy[1] - y],
                              mem])

        self.model.body_quat[21] = my_utils.rpy_to_quat(0, 0, yaw)
        self.model.body_pos[21] = [x,y,1]

        return obs, r, done, obs_dict


    def reset(self, test=False):
        self.step_ctr = 0

        #for i in range(6):
        #    self.model.geom_rgba[self.model._geom_name2id[self.leg_list[i]]] = [0.0, 0.6, 0.4, 1]

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = np.random.randn() * 0.1 + 0.05
        init_q[1] = np.random.randn() * 0.1
        init_q[2] = 0.2
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        # Set environment state
        self.set_state(init_q, init_qvel)

        self.goal_xy = np.random.randn(2) * 1.0

        self.model.body_pos[20] = [*self.goal_xy, 0]

        x, y, z, q0, q1, q2, q3 = self.get_obs()[:7]

        obs_dict = self.get_obs_dict()
        obs = np.concatenate([self.sim.get_state().qpos.tolist()[2:],
                              self.sim.get_state().qvel.tolist(),
                              obs_dict["contacts"],
                              [self.goal_xy[0] - x, self.goal_xy[1] - y],
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
            obs = self.reset(test=True)
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
                obs, r, done, od, = self.step(action[0].detach())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))

if __name__ == "__main__":
    ant = Hexapod(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()