import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from driver import Driver
from ax12 import *

class NN_PG(nn.Module):
    def __init__(self, env, hid_dim=64, tanh=False, std_fixed=True, obs_dim=None, act_dim=None):
        super(NN_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        if obs_dim is not None:
            self.obs_dim = obs_dim

        if act_dim is not None:
            self.act_dim = act_dim

        self.tanh = tanh

        self.fc1 = nn.Linear(self.obs_dim, hid_dim)
        self.m1 = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.m2 = nn.LayerNorm(hid_dim)
        self.fc3 = nn.Linear(hid_dim, self.act_dim)

        T.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='linear')


    def forward(self, x):
        x = F.leaky_relu(self.m1(self.fc1(x)))
        x = F.leaky_relu(self.m2(self.fc2(x)))
        if self.tanh:
            x = T.tanh(self.fc3(x))
        else:
            x = self.fc3(x)
        return x


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


class HexapodController:
    def __init__(self, policy_path=None):
        self.policy_path = policy_path

        if self.policy_path is not None:
            logging.info("Loading policy: \"{}\" ".format(self.policy_path))
            self.nn_policy = T.load(self.policy_path)

        logging.info("Initializing robot hardware")
        if not self.init_hardware():
            logging.error("Robot hardware communication issue, exiting")
            exit()

        logging.info("Starting control loop")
        while True:
            # Read robot servos and hardware and turn into observation for nn
            policy_obs = self.hex_get_obs()

            # Perform forward pass on nn policy
            policy_act = self.nn_policy(policy_obs)

            # Calculate servo commands from policy action and write to servos
            self.hex_write_ctrl(policy_act)


    def init_hardware(self):
        '''
        Initialize robot hardware and variables
        :return: Boolean
        '''


        self.driver = Driver(port='/dev/ttyUSB0')

        # Initialize variables for servos
        self.servo_positions = [None] * 18
        self.servo_torques = [None] * 18
        self.servo_goal_positions = [None] * 18

        # Set servo parameters
        self.max_servo_speed = 700 # [0:1024]
        self.max_servo_torque = 700 # [0:1024]

        self.driver.setReg(1, P_GOAL_SPEED_L, [self.max_servo_speed % 256, self.max_servo_speed >> 8])
        self.driver.setReg(1, P_MAX_TORQUE_L, [self.max_servo_torque % 256, self.max_servo_torque >> 8])

        self.rob_to_policy_servo_ID = [1,3,5, ]
        self.policy_to_rob_servo_ID = {}

        return True

    def _make_obs_for_nn(self):
        '''
        Turn robot observation into observation for policy input
        :return: obs vector
        '''
        pass

    def _make_act_for_hex(self, policy_act):
        '''
        Turn policy action into servo commands
        :return: hardware packet
        '''
        pass

    def hex_get_obs(self):
        '''
        Read robot hardware and return observation
        :return:
        '''

        is_moving = self.driver.getReg(1, P_MOVING, 1)
        pass

    def hex_write_ctrl(self, nn_act):
        '''
        Turn policy action into servo commands and write them to servos
        :return: None
        '''
        pass
