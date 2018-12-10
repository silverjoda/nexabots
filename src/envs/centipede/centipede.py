import numpy as np
import pybullet as p
import pybullet_data
from collections import deque
import time


class Centipede8:
    def __init__(self, GUI=True):

        # Start client
        physicsClient = p.connect(p.GUI if GUI else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set simulation parameters
        p.setGravity(0, 0, -9.81)

        # This is required so that we step simulation
        p.setRealTimeSimulation(0)

        # Simulation time step (Not recommended to change (?))
        # p.setTimeStep(0.0001)

        # Load simulator objects
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadMJCF("CentipedeEight.xml")[0]

        self.joint_dict = {}
        for i in range(p.getNumJoints(self.robotId)):
            info = p.getJointInfo(self.robotId, i)
            id, name, joint_type = info[0:3]
            joint_lim_low, joint_lim_high = info[8:10]
            if joint_type == 0:
                self.joint_dict[name] = (id, joint_lim_low, joint_lim_high)
        self.joint_ids = [j[0] for j in self.joint_dict.values()]
        self.n_joints = len(self.joint_ids)

        self.obs_dim = 7 + 6 + self.n_joints * 2
        self.act_dim = self.n_joints

        # Environent inner parameters
        self.success_queue = deque(maxlen=100)
        self.viewer = None
        self.goal = None
        self.current_pose = None
        self.success_rate = 0
        self.step_ctr = 0

        # Initial methods
        self.reset()
        self.setupcam()


    def setupcam(self):
        pass


    def get_obs(self):
        # Base position and orientation
        torso_pose = p.getBasePositionAndOrientation(self.robotId)

        # Base velocity and angular velocity
        torso_pose_ = p.getBaseVelocity(self.robotId)

        # Joint states and velocities
        q, q_, _, _ = zip(*p.getJointStates(self.robotId, self.joint_ids))

        obs_dict = {'torso_pos': torso_pose[0:3],
                    'torso_quat': torso_pose[3:7],
                    'q': q,
                    'torso_vel': torso_pose_[0:3],
                    'torso_angvel': torso_pose_[3:6],
                    'q_vel': q_}

        # TODO: Error here, torso_angvel is empty and error in concat below

        obs_arr = np.concatenate([v for v in obs_dict.values()])

        return obs_arr, obs_dict


    def step(self, ctrl):

        p.setJointMotorControlArray(self.robotId, self.joint_ids, p.TORQUE_CONTROL, ctrl)

        self.step_ctr += 1
        obs_arr, obs_dict = self.get_obs()

        # Make relevant pose from observation (x,y)
        x, y = obs_dict["torso_pos"][0:2]
        pose = (x, y)

        prev_dist = np.sqrt(np.sum((np.asarray(self.current_pose) - np.asarray(self.goal)) ** 2))
        current_dist = np.sqrt(np.sum((np.asarray(pose) - np.asarray(self.goal)) ** 2))

        # Reevaluate termination condition
        done = self.step_ctr > 400

        ctrl_effort = np.square(ctrl).sum() * 0.01
        target_progress = (prev_dist - current_dist) * 70

        r = target_progress - ctrl_effort

        self.current_pose = pose

        return obs_arr, r, done, obs_dict


    def reset(self):

        # Reset env variables
        self.step_ctr = 0

        # Variable positions
        obs_dict = {'torso_pos' : (0, 0, 0.6),
                    'torso_quat' : (0, 0, 0, 1),
                    'q' : np.zeros(self.n_joints),
                    'torso_vel' : np.zeros(self.n_joints),
                    'torso_angvel' : np.zeros(6),
                    'q_vel' : np.zeros(self.n_joints)}

        obs_arr = np.concatenate([v for v in obs_dict.values()])

        # Set environment state
        p.resetBasePositionAndOrientation(self.robotId, obs_dict['torso_pos'], obs_dict['torso_quat'])
        p.resetBaseVelocity(self.robotId, obs_dict['torso_vel'])

        for j in self.joint_ids:
            p.resetJointState(self.robotId, j, 0, 0)

        return obs_arr, obs_dict


    def demo(self):
        self.reset()
        for i in range(1000):
            self.step(np.random.randn(self.n_joints))
            self.render()


    def random_action(self):
        return np.random.randn(self.n_joints)


if __name__ == "__main__":
    cent = Centipede8()
    print(cent.obs_dim)
    print(cent.act_dim)
    cent.demo()
