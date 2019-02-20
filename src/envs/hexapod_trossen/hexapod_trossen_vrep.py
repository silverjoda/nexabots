import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
from math import sqrt, acos, fabs
from src.envs.hexapod_terrain_env.hf_gen import ManualGen, EvoGen, HMGen
import random
import string

#hexapod robot
import src.envs.hexapod_trossen.hexapod_sim.RobotHAL as robothal
#import hexapod_real.RobotHAL as robothal

from src.envs.hexapod_trossen.RobotConst import *
import src.policies as policies

import torch as T

class Hexapod:

    def __init__(self, animate=False):

        self.mem_dim = 0

        self.joints_rads_low = np.array([-0.6, -1., -1.] * 6)
        self.joints_rads_high = np.array([0.6, 0.3, 1.] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.robot = robothal.RobotHAL(TIME_FRAME, True)

        self.obs_dim = 18 + 3 + 6
        self.act_dim = 18
        self.sign_corr_vec = np.array([1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1])

        # Environent inner parameters
        self.viewer = None

        # Reset env variables
        self.step_ctr = 0
        self.episodes = 0

        self.prev_servo_pos = np.zeros(18)

        self.turn_on()
        time.sleep(3)

        # TODO: Test the action and observation sign and position correctness
        # TODO: Check the correspondant angles in MUJOCO as well to make sure that assumption was correct

    def turn_on(self):
        """
        Method to drive the robot into the default position
        """
        # read out the current pose of the robot
        pose = self.robot.get_all_servo_position()

        # interpolate to the default position
        INTERPOLATION_TIME = 3000  # ms
        interpolation_steps = int(INTERPOLATION_TIME / TIME_FRAME)

        speed = np.zeros(18)
        for i in range(0, 18):
            speed[i] = (SERVOS_BASE[i] - pose[i]) / interpolation_steps

        # execute the motion
        for t in range(0, 1):
            self.robot.set_all_servo_position(pose + t * speed)
            pass



    def mj_to_vrep(self, joints):
        return [joints[0], #1
                joints[3], #2
                joints[1], #3
                joints[4], #4
                joints[2], #5
                joints[5], #6
                joints[9], #7
                joints[6], #8
                joints[10], #9
                joints[7], #10
                joints[11], #11
                joints[8], #12
                joints[15], #13
                joints[12], #14
                joints[16], #15
                joints[13], #16
                joints[17], #17
                joints[14], #18
                ]


    def vrep_to_mj(self, joints):
        return [joints[0],   # 1 # FL
                joints[2],   # 2
                joints[4],   # 3
                joints[12],   # 4 # ML
                joints[14],   # 5
                joints[16],   # 6
                joints[1],   # 7 # FR
                joints[3],   # 8
                joints[5],  # 9
                joints[13],   # 10 # MR
                joints[15],  # 11
                joints[17],   # 12
                joints[7],  # 13 # RR
                joints[9],  # 14
                joints[11],  # 15
                joints[6],  # 16 # RL
                joints[8],  # 17
                joints[10],  # 18
                ]


    def correct_sign(self, joints):
        return np.array(joints) * self.sign_corr_vec


    def turn_off(self):
        self.robot.stop_simulation()


    def scale_action(self, action):
        a = (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low
        return np.clip(a, self.joints_rads_low, self.joints_rads_high)



    def get_obs(self):
        # Joint angles
        q = self.robot.get_all_servo_position()
        q = self.correct_sign(q)
        q = self.vrep_to_mj(q)

        # Orientation
        rot = self.robot.get_robot_orientation()
        if rot is None:
            rot = [0,0,0]

        # Contacts
        contacts = self.robot.get_leg_contacts()

        return np.concatenate([np.array(q), np.array(rot), np.array(contacts)])


    def step(self, ctrl):
        ctrl = self.scale_action(ctrl)
        ctrl = self.mj_to_vrep(ctrl)
        ctrl = self.correct_sign(ctrl)

        self.robot.set_all_servo_position(ctrl)
        self.step_ctr += 1

        obs = self.get_obs()

        return obs, None, None, None


    def reset(self, test=False):
        self.cumulative_environment_reward = 0
        self.step_ctr = 0
        return self.get_obs()


    def demo(self, policy):
        obs = self.reset()
        for i in range(1000):
            act = policy(my_utils.to_tensor(obs, True))[0].detach().numpy()
            obs, _, _, _ = self.step(act)
            print(i)


    def info(self):
        self.reset()
        for i in range(100):
            a = np.ones((self.act_dim)) * 0
            a[[15,16,17]] = -1
            obs, _, _, _ = self.step(a)
            print(obs[[3,4,5]])

        print("-------------------------------------------")
        print("-------------------------------------------")



if __name__ == "__main__":
    pod = Hexapod(animate=True)

    #policy = policies.NN_PG(pod)
    policy = T.load('/home/silverjoda/PycharmProjects/nexabots/src/algos/PG/agents/Hexapod_NN_PG_8AX_pg.p')
    print(pod.obs_dim)
    print(pod.act_dim)
    pod.demo(policy)
