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

        self.leg_servo_indeces = [self.policy_to_servo_mapping[i*3:i*3+3] for i in range(6)]

        return True


    def test_leg_coordination(self):
        '''
        Perform leg test to determine correct mapping and range
        :return:
        '''
        pass


    def _policy_to_servo(self, vec):
        return [vec[self.policy_to_servo_mapping[i]] for i in range(18)]


    def _servo_to_policy(self, vec):
        return [vec[self.servo_to_policy_mapping[i]] for i in range(18)]


    def hex_get_obs(self):
        '''
        Read robot hardware and return observation
        :return:
        '''

        # Read servo data
        self.servo_positions = [self.driver.getReg(i, P_PRESENT_POSITION_L, 1) for i in range(18)]
        self.servo_torques = [self.driver.getReg(i, P_PRESENT_LOAD_L, 1) for i in range(18)]

        # Read IMU (for now spoof perfect orientation)
        self.yaw = 0

        # Calculate leg contact
        self._infer_legtip_contact()

        # Make nn observation
        obs = None

        return obs


    def hex_write_ctrl(self, nn_act):
        '''
        Turn policy action into servo commands and write them to servos
        :return: None
        '''

        # Map correct actuator to servo
        servo_act = np.array(self._policy_to_servo(nn_act))

        # Map [-1,1] to [0, 1024]
        servo_act = ((servo_act + 1) * 0.5) * self.servo_range + self.servo_low

        statuses = [self.driver.setReg(1, P_GOAL_POSITION_L, [servo_act[i] % 256, nn_act >> 8])]

        return max(statuses)


    def _infer_legtip_contact(self):
        def _leg_is_in_contact(servo_vec):
            return (servo_vec[1] > 100 and servo_vec[2] > 100)

        self.legtip_contact_vec = [_leg_is_in_contact(self.leg_servo_indeces[i]) for i in range(3)]




