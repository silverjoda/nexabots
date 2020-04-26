import time
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

        # Parameters
        self.servo_low, self.servo_high = 256, 768
        self.servo_range = self.servo_high - self.servo_low

        self.policy_path = policy_path
        if self.policy_path is not None:
            logging.info("Loading policy: \"{}\" ".format(self.policy_path))
            self.nn_policy = T.load(self.policy_path)

        logging.info("Initializing robot hardware")
        if not self.init_hardware():
            logging.error("Robot hardware communication issue, exiting")
            exit()

        # Start control loop
        self.start_ctrl_loop()


    def start_ctrl_loop(self):
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
        self.legtip_contact_vec = [None] * 6

        # Set servo parameters
        self.max_servo_speed = 700 # [0:1024]
        self.max_servo_torque = 700 # [0:1024]

        self.driver.setReg(1, P_GOAL_SPEED_L, [self.max_servo_speed % 256, self.max_servo_speed >> 8])
        self.driver.setReg(1, P_MAX_TORQUE_L, [self.max_servo_torque % 256, self.max_servo_torque >> 8])

        self.policy_to_servo_mapping = [1, 3, 5, 13, 15, 17, 2, 4, 6, 14, 16, 18, 8, 10, 12, 7, 9, 11]
        self.servo_to_policy_mapping = [self.policy_to_servo_mapping.index(i + 1) for i in range(18)]

        self.joints_rads_low = np.array([-0.4, -1.0, -0.5] * 6)
        self.joints_rads_high = np.array([0.4, 0.0, 0.5] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.joints_10bit_low = ((self.joints_rads_low) / (5.23599) + 0.5) * 1024
        self.joints_10bit_high = ((self.joints_rads_high) / (5.23599) + 0.5) * 1024
        self.joints_10bit_diff = self.joints_10bit_high - self.joints_10bit_low

        self.leg_servo_indeces = [self.policy_to_servo_mapping[i*3:i*3+3] for i in range(6)]

        return True


    def test_leg_coordination(self):
        '''
        Perform leg test to determine correct mapping and range
        :return:
        '''

        logging.info("Starting leg coordination test")

        self.hex_write_ctrl(T.zeros((18)))
        time.sleep(5)

        self.hex_write_ctrl(T.tensor([0., -1., 1.] * 6))
        time.sleep(5)

        self.hex_write_ctrl(T.tensor([0., -1., 1.] * 6))
        time.sleep(5)

        self.hex_write_ctrl(T.ones(18) * 1)
        time.sleep(5)

        self.hex_write_ctrl(T.ones(8) * - 1)
        time.sleep(5)

        logging.info("Finished leg coordination test")


    def hex_get_obs(self):
        '''
        Read robot hardware and return observation tensor for pytorch
        :return:
        '''

        # Read servo data
        self.servo_positions = [self.driver.getReg(i, P_PRESENT_POSITION_L, 1) for i in range(18)]

        # Read IMU (for now spoof perfect orientation)
        self.yaw = 0

        # Map servo inputs to correct NN inputs
        mapped_servo_positions = np.array([self.servo_positions[self.servo_to_policy_mapping[i]] for i in range(18)])

        # Turn servo positions into [-1,1] for nn
        scaled_nn_actuator_positions = ((mapped_servo_positions - self.joints_10bit_low) / self.joints_10bit_diff) * 2 - 1

        # Make nn observation
        obs = np.concatenate((scaled_nn_actuator_positions, [0]))

        # Make pytorch tensor from observation
        t_obs = T.tensor(obs).unsqueeze(0)

        return t_obs


    def hex_write_ctrl(self, nn_act):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''

        # Map [-1,1] to correct 10 bit servo value, respecting the scaling limits imposed during training
        scaled_act = (nn_act[0].numpy() * 0.5 + 0.5) * self.joints_10bit_diff + self.joints_10bit_low

        # Map correct actuator to servo
        servo_act = np.array([scaled_act[self.policy_to_servo_mapping[i]] for i in range(18)])

        # Write commands to servos and read error statuses
        statuses = [self.driver.setReg(1, P_GOAL_POSITION_L, [servo_act[i] % 256, nn_act >> 8]) for i in range(18)]

        return max(statuses)


    def _infer_legtip_contact(self):
        def _leg_is_in_contact(servo_vec):
            return (servo_vec[1] > 100 and servo_vec[2] > 100)

        self.legtip_contact_vec = [_leg_is_in_contact(self.leg_servo_indeces[i]) for i in range(3)]




