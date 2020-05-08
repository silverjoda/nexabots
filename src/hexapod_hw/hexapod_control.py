import time
import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from src.hexapod_hw.driver import Driver
from src.hexapod_hw.ax12 import *

class NN_PG(nn.Module):
    def __init__(self, hid_dim=64, tanh=False, std_fixed=True, obs_dim=None, act_dim=None):
        super(NN_PG, self).__init__()

        self.obs_dim = obs_dim
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

        # Set servo parameters
        self.max_servo_speed = 500  # [0:1024]
        self.max_servo_torque = 800  # [0:1024]

        self.policy_to_servo_mapping = [1, 3, 5, 2, 4, 6, 8, 10, 12, 7, 9, 11, 14, 16, 18, 13, 15, 17]
        self.servo_to_policy_mapping = [0, 6, 1, 7, 2, 8, 15, 12, 16, 13, 17, 14, 3, 9, 4, 10, 5, 11]

        self.joints_rads_low = np.array([-0.3, -1.4, 0.6] * 6)
        self.joints_rads_high = np.array([0.3, 0.0, 0.9] * 6)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.joints_10bit_low = ((self.joints_rads_low) / (5.23599) + 0.5) * 1024
        self.joints_10bit_high = ((self.joints_rads_high) / (5.23599) + 0.5) * 1024
        self.joints_10bit_diff = self.joints_10bit_high - self.joints_10bit_low

        self.leg_servo_indeces = [self.policy_to_servo_mapping[i * 3:i * 3 + 3] for i in range(6)]

        self.policy_path = policy_path
        if self.policy_path is not None:
            logging.info("Loading policy: \"{}\" ".format(self.policy_path))
            self.nn_policy = NN_PG(96, obs_dim = 19, act_dim=18)
            self.nn_policy.load_state_dict(T.load(self.policy_path))


        logging.info("Initializing robot hardware")
        if not self.init_hardware():
            logging.error("Robot hardware communication issue, exiting")
            exit()


    def testcom(self):
        for i in range(100):
            t1 = time.time()
            _ = [self.driver.getReg(i + 1, P_PRESENT_POSITION_L, 1) for i in range(18)]
            t2 = time.time()
            logging.info("Reading position register from servos took: {} s".format(t2-t1))



    def start_ctrl_loop(self):
        logging.info("Starting control loop")
        while True:
            # Read robot servos and hardware and turn into observation for nn
            t1 = time.time()
            policy_obs = self.hex_get_obs()
            t2 = time.time()
            print("obs", t2-t1)

            # Perform forward pass on nn policy
            policy_act = self.nn_policy(policy_obs)

            # Calculate servo commands from policy action and write to servos
            t1 = time.time()
            self.hex_write_ctrl(policy_act)
            t2 = time.time()
            print("act", t2-t1)

    def init_hardware(self):
        '''
        Initialize robot hardware and variables
        :return: Boolean
        '''

        self.driver = Driver(port='/dev/ttyUSB0', baud=115200)

        #is_moving = self.driver.getReg(1, P_MOVING, 1)
        statuses_speed = [self.driver.setReg(i + 1, P_GOAL_SPEED_L, [self.max_servo_speed % 256, self.max_servo_speed >> 8]) for i in range(18)]

        statuses_torque = [self.driver.setReg(i + 1, P_MAX_TORQUE_L, [self.max_servo_torque % 256, self.max_servo_torque >> 8]) for i in range(18)]

        statuses_total = statuses_speed + statuses_torque
    
        if max(statuses_total) > 0:
            logging.warn("Hardware init possible fault")

        #self.hex_write_ctrl(T.zeros(18).unsqueeze(0))                
        self.hex_write_servos_direct([512]*18)
        time.sleep(1.5)

        return True 


    def test_leg_coordination(self):
        '''
        Perform leg test to determine correct mapping and range
        :return:
        '''

        logging.info("Starting leg coordination test")
            

        status = self.hex_write_ctrl(T.zeros((18)))
        logging.info("Testing leg coordination, status: {}".format(status))
        time.sleep(5)

        status = self.hex_write_ctrl(T.tensor([0., -1., 1.] * 6))
        logging.info("Testing leg coordination, status: {}".format(status))
        time.sleep(5)

        status = self.hex_write_ctrl(T.tensor([0., 1., -1.] * 6))
        logging.info("Testing leg coordination, status: {}".format(status))
        time.sleep(5)

        status = self.hex_write_ctrl(T.ones(18) * 1)
        logging.info("Testing leg coordination, status: {}".format(status))
        time.sleep(5)

        status = self.hex_write_ctrl(T.ones(18) * - 1)
        logging.info("Testing leg coordination, status: {}".format(status))
        time.sleep(5)

        logging.info("Finished leg coordination test")


    def hex_get_obs(self):
        '''
        Read robot hardware and return observation tensor for pytorch
        :return:
        '''

        # Read servo data
        servo_reg_L = [self.driver.getReg(i+1, P_PRESENT_POSITION_L, 1) for i in range(18)]
        servo_reg_H = [self.driver.getReg(i+1, P_PRESENT_POSITION_H, 1) for i in range(18)]
        self.servo_positions = np.array([servo_reg_L[i] + (servo_reg_H[i] << 8)  for i in range(18)]).astype(np.float32)

        self.servo_positions[np.array([4,6,16,18,10,12])-1] = 1024 - self.servo_positions[np.array([4,6,16,18,10,12])-1]  
        # Read IMU (for now spoof perfect orientation)
        self.yaw = 0

        # Map servo inputs to correct NN inputs
        mapped_servo_positions = np.zeros((18))
        for i in range(18):
            mapped_servo_positions[self.servo_to_policy_mapping[i]] = self.servo_positions[i]

        # Turn servo positions into [-1,1] for nn
        scaled_nn_actuator_positions = ((mapped_servo_positions - self.joints_10bit_low) / self.joints_10bit_diff) * 2 - 1

        # Make nn observation
        obs = np.concatenate((scaled_nn_actuator_positions, [0])).astype(np.float32)
        logging.info("NN observation: {}".format(obs))

        # Make pytorch tensor from observation
        t_obs = T.tensor(obs).unsqueeze(0)

        return t_obs


    def hex_write_ctrl(self, nn_act):
        '''
        Turn policy action tensor into servo commands and write them to servos
        :return: None
        '''

        #logging.info("Servo action: {}".format(nn_act))

        # Map [-1,1] to correct 10 bit servo value, respecting the scaling limits imposed during training
        scaled_act = [(np.asscalar(nn_act[0][i].detach().numpy()) * 0.5 + 0.5) * self.joints_10bit_diff[i] + self.joints_10bit_low[i] for i in range(18)]

        # Map correct actuator to servo
        servo_act = np.zeros((18), dtype=np.uint16)
        for i in range(18):
            servo_act[self.policy_to_servo_mapping[i] - 1] = scaled_act[i]

        # Reverse servo signs for right hand servos (Thsi part is retarded and will need to be fixed)
        servo_act[np.array([4,6,16,18,10,12])-1] = 1024 - servo_act[np.array([4,6,16,18,10,12])-1]  
            
        # Write commands to servos and read error statuses
        statuses = [self.driver.setReg(i+1, P_GOAL_POSITION_L, [servo_act[i] % 256, servo_act[i] >> 8]) for i in range(18)]

        time.sleep(0.1)

        return max(statuses)


    def hex_write_servos_direct(self, act):
        statuses = [self.driver.setReg(i+1, P_GOAL_POSITION_L, [act[i] % 256, act[i] >> 8]) for i in range(18)]
        return max(statuses)


    def _infer_legtip_contact(self):
        def _leg_is_in_contact(servo_vec):
            return (servo_vec[1] > 100 and servo_vec[2] > 100)

        self.legtip_contact_vec = [_leg_is_in_contact(self.leg_servo_indeces[i]) for i in range(3)]


if __name__ == "__main__":
    controller = HexapodController("Hexapod_NN_PG_LEP_pg.p")
    #controller.testcom()
    #controller.start_ctrl_loop()
