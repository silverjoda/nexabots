import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
import cv2
from src.envs.locom_benchmarks import hf_gen
import gym
from gym import spaces
from math import acos

class Hexapod(gym.Env):
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "hex_test.xml")

    def __init__(self):
        print("Hexapod flat")

        # External parameters
        self.joints_rads_low = np.array([-0.6, -1.4, -0.8] * 6)
        self.joints_rads_high = np.array([0.6, 0.4, 0.8] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.reset()

        self.observation_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.obs_dim,))
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.act_dim,))

        self.setupcam()


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


    def scale_action(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low


    def step(self, ctrl):
        # Clip control signal
        ctrl = np.clip(ctrl, -1, 1)

        # Control penalty
        ctrl_pen = np.square(ctrl).mean()

        # Scale control according to joint ranges
        ctrl = self.scale_action(ctrl)

        # TODO: Continue building the first environment, then copy to all the rest of the blind ones, then do rangefinder and feelers, then heterogeneous environments, then camera

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        obs = self.get_obs()

        # Angle deviation
        x, y, z, qw, qx, qy, qz = obs[:7]
        xd, yd, zd, thd, phid, psid = self.sim.get_state().qvel.tolist()[:6]
        # xa, ya, za, tha, phia, psia = self.sim.data.qacc.tolist()[:6]

        self.vel_sum += xd

        # Reward conditions
        target_vel = 0.4
        # avg_vel = self.vel_sum / self.step_ctr

        velocity_rew = 1. / (abs(xd - target_vel) + 1.) - 1. / (target_vel + 1.)
        velocity_rew = velocity_rew * (1 / (1 + 30 * np.square(yd)))

        roll, pitch, yaw = my_utils.quat_to_rpy([qw, qx, qy, qz])
        yaw_deviation = np.min((abs((yaw % 6.183) - (0 % 6.183)), abs(yaw - 0)))

        q_yaw = 2 * acos(qw)

        r_neg = np.square(y) * 0.2 + \
                np.square(q_yaw) * 0.5 + \
                np.square(pitch) * 0.5 + \
                np.square(roll) * 0.5 + \
                np.square(ctrl_pen) * 0.1 + \
                np.square(zd) * 0.7

        r_pos = velocity_rew * 6  # + (abs(self.prev_yaw_deviation) - abs(yaw_deviation)) * 3. + (abs(self.prev_y_deviation) - abs(y)) * 3.
        r = r_pos - r_neg

        self.prev_yaw_deviation = yaw_deviation
        self.prev_y_deviation = y

        # Reevaluate termination condition
        done = self.step_ctr > self.max_steps  # or abs(y) > 0.3 or abs(yaw) > 0.6 or abs(roll) > 0.8 or abs(pitch) > 0.8
        contacts = (np.abs(np.array(self.sim.data.cfrc_ext[[4, 7, 10, 13, 16, 19]])).sum(axis=1) > 0.05).astype(
            np.float32)


        obs = np.concatenate([self.scale_joints(self.sim.get_state().qpos.tolist()[7:]),
                              self.sim.get_state().qvel.tolist()[6:],
                              self.sim.get_state().qvel.tolist()[:6],
                              [qw, qx, qy, qz, y],
                              contacts])

        return obs, r, done, None


    def reset(self):
        # Generate environment
        hm = hf_gen.hm_flat(1)
        cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hm.png"), hm)

        # Load simulator
        self.model = mujoco_py.load_model_from_path(Hexapod.MODELPATH)
        self.sim = mujoco_py.MjSim(self.model)
        self.model.opt.timestep = 0.02
        self.viewer = None

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = 18 * 2 + 4 + 6 + 6 # j, jd, quat, pose_velocity, contacts
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Set initial position
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = 0.0
        init_q[1] = 0.0
        init_q[2] = 0.50
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        # Set environment state
        self.set_state(init_q, init_qvel)

        self.step_ctr = 0

        obs, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs


    def render(self, camera=None):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()


    def demo(self):
        for i in range(100000):
            self.sim.forward()
            self.sim.step()
            self.render()


if __name__ == "__main__":
    hex = Hexapod()
    hex.demo()