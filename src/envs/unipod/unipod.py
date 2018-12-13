import numpy as np
import pybullet as p
import pybullet_data
from collections import deque
import time
import os
import src.my_utils as my_utils

class Unipod:
    def __init__(self, gui=False):

        # Start client
        physicsClient = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set simulation parameters
        p.setGravity(0, 0, -9.81)

        p.setRealTimeSimulation(0)

        # Simulation time step (Not recommended to change (?))
        #p.setTimeStep(0.0001)

        # Load simulator objects
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF(os.path.join(os.path.dirname(__file__), "unipod.urdf"))
        for i in range(1000000):
            p.stepSimulation()
            p.setJointMotorControl2(self.robotId, 0, p.POSITION_CONTROL, targetPosition=1)

            time.sleep(0.001)
        exit()

        self.joint_dict = {}
        for i in range(p.getNumJoints(self.robotId)):
            info = p.getJointInfo(self.robotId, i)
            id, name, joint_type = info[0:3]#
            joint_lim_low, joint_lim_high = info[8:10]
            self.joint_dict[name] = (id, joint_lim_low, joint_lim_high)


        self.joint_ids = [j[0] for j in self.joint_dict.values()]
        self.n_joints = len(self.joint_ids)

        self.obs_dim = 7 + 6 + self.n_joints * 2
        self.act_dim = self.n_joints

        # Environent inner parameters
        self.step_ctr = 0


        # Initial methods
        self.reset()
        self.setupcam()


    def setupcam(self):
        pass


    def get_obs(self):
        # Base position and orientation
        torso_pos, torso_orient = p.getBasePositionAndOrientation(self.robotId)

        # Base velocity and angular velocity
        torso_pos_, torso_orient_ = p.getBaseVelocity(self.robotId)

        # Joint states and velocities
        q, q_, _, _ = zip(*p.getJointStates(self.robotId, self.joint_ids))

        obs_dict = {'torso_pos': torso_pos,
                    'torso_quat': torso_orient,
                    'q': q,
                    'torso_vel': torso_pos_,
                    'torso_angvel': torso_orient_,
                    'q_vel': q_}

        obs_arr = np.concatenate([v for v in obs_dict.values()]).astype(np.float32)

        return obs_arr, obs_dict


    def step(self, ctrl):

        # Get measurements before step
        torso_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        x_prev = torso_pos[0]

        # Add control law
        p.setJointMotorControlArray(self.robotId, self.joint_ids, p.TORQUE_CONTROL, forces=ctrl * 1500)

        # Perform single step of simulation
        p.stepSimulation()

        # Get measurements after step
        torso_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        x_current = torso_pos[0]

        self.step_ctr += 1
        obs_arr, obs_dict = self.get_obs()

        # Reevaluate termination condition
        done = self.step_ctr > 400

        ctrl_effort = np.square(ctrl).mean() * 0.01
        target_progress = (x_current - x_prev) * 50

        r = target_progress - ctrl_effort

        return obs_arr, r, done, obs_dict


    def reset(self):

        # Set zero torques
        p.setJointMotorControlArray(self.robotId, self.joint_ids, p.TORQUE_CONTROL, forces=[0] * self.act_dim)
        p.stepSimulation()

        # Reset env variables
        self.step_ctr = 0

        # Variable positions
        obs_dict = {'torso_pos' : (0, 0, 0.7),
                    'torso_quat' : (0, 0, 0, 1),
                    'q' : np.zeros(self.n_joints),
                    'torso_vel' : np.zeros(3),
                    'torso_angvel' : np.zeros(3),
                    'q_vel' : np.zeros(self.n_joints)}

        obs_arr = np.concatenate([v for v in obs_dict.values()])

        # Set environment state
        p.resetBasePositionAndOrientation(self.robotId, obs_dict['torso_pos'], obs_dict['torso_quat'])
        p.resetBaseVelocity(self.robotId, obs_dict['torso_vel'])

        for j in self.joint_ids:
            p.resetJointState(self.robotId, j, 0, 0)

        p.setJointMotorControlArray(self.robotId, self.joint_ids, p.TORQUE_CONTROL, forces=[0] * self.act_dim)
        for i in range(30):
            p.stepSimulation()

        return obs_arr, obs_dict


    def demo(self):
        self.reset()
        t1 = time.time()
        iters = 10000
        for i in range(iters):
            if i % 1000 == 0:
                print("Step: {}/{}".format(i, iters))
                self.reset()
            self.step(np.random.randn(self.n_joints))
        t2 = time.time()
        print("Time Elapsed: {}".format(t2-t1))


    def test(self, policy):
        self.reset()
        for i in range(100):
            done = False
            obs, _ = self.reset()
            cr = 0
            while not done:
                action = policy(my_utils.to_tensor(obs, True)).detach()
                obs, r, done, od, = self.step(action[0])
                cr += r
                time.sleep(0.001)
            print("Total episode reward: {}".format(cr))


    def random_action(self):
        return np.random.randn(self.n_joints)


if __name__ == "__main__":
    upod = Unipod(gui=True)
    print(upod.obs_dim)
    print(upod.act_dim)
    upod.demo()
