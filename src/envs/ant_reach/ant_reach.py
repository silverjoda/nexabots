import numpy as np
import pybullet as p
import pybullet_data
from collections import deque
import time
import os
import src.my_utils as my_utils

class AntReach:
    def __init__(self, gui=False):

        # Start client
        self.physicsClient = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set simulation parameters
        p.setGravity(0, 0, -9.81)

        # This is required so that we step simulation
        p.setRealTimeSimulation(0)

        # Simulation time step (Not recommended to change (?))
        # p.setTimeStep(0.0001)

        # Load simulator objects
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId, self.targetId = p.loadMJCF(os.path.join(os.path.dirname(__file__), "ant_reach.xml"))

        self.joint_lim_lows = []
        self.joint_lim_highs = []
        self.joint_dict = {}
        for i in range(p.getNumJoints(self.robotId)):
            info  = p.getJointInfo(self.robotId, i)
            id, name, joint_type = info[0:3]
            joint_lim_low, joint_lim_high = info[8:10]
            if joint_type == 0:
                self.joint_lim_lows.append(joint_lim_low)
                self.joint_lim_highs.append(joint_lim_high)
                self.joint_dict[name] = (id, joint_lim_low, joint_lim_high)

        self.joint_lim_lows = np.asarray(self.joint_lim_lows)
        self.joint_lim_highs = np.asarray(self.joint_lim_highs)

        self.joint_ids = [j[0] for j in self.joint_dict.values()]
        self.n_joints = len(self.joint_ids)

        self.obs_dim = 7 + 6 + self.n_joints * 2 + 2 # Last 3 are target x,y
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
        torso_pos, torso_orient = p.getBasePositionAndOrientation(self.robotId)

        # Target position and orientation
        target_pos, _ = p.getBasePositionAndOrientation(self.targetId)

        # Base velocity and angular velocity
        torso_pos_, torso_orient_ = p.getBaseVelocity(self.robotId)

        # Joint states and velocities
        q, q_, _, _ = zip(*p.getJointStates(self.robotId, self.joint_ids))

        obs_dict = {'torso_pos': torso_pos,
                    'torso_quat': torso_orient,
                    'target_pos': target_pos[0:2],
                    'q': q,
                    'torso_vel': torso_pos_,
                    'torso_angvel': torso_orient_,
                    'q_vel': q_}

        obs_arr = np.concatenate([v for v in obs_dict.values()]).astype(np.float32)

        return obs_arr, obs_dict


    def _sample_goal(self, pose):
        while True:
            x, y = pose
            nx = x + np.random.randn() * (2. + 3 * self.success_rate)
            ny = y + np.random.randn() * (2. + 3 * self.success_rate)

            goal = nx, ny

            if not self.reached_goal(pose, goal):
                break

        return np.array(goal)


    def _update_stats(self, reached_goal):
        self.success_queue.append(1. if reached_goal else 0.)
        self.success_rate = np.mean(self.success_queue)


    def reached_goal(self, pose, goal):
        x,y = pose
        xg,yg = goal
        return (x-xg)**2 < 0.2 and (y-yg)**2 < 0.2


    def step(self, ctrl):

        # Get measurements before step
        torso_pos_prev, _ = p.getBasePositionAndOrientation(self.robotId)

        # Add control law
        p.setJointMotorControlArray(self.robotId, self.joint_ids, p.TORQUE_CONTROL, forces=ctrl * 1300)

        # Perform single step of simulation
        p.stepSimulation()

        # Get measurements after step
        torso_pos_current, _ = p.getBasePositionAndOrientation(self.robotId)

        prev_dist = np.sqrt(np.sum((np.asarray(torso_pos_prev[0:2]) - np.asarray(self.goal)) ** 2))
        current_dist = np.sqrt(np.sum((np.asarray(torso_pos_current[0:2]) - np.asarray(self.goal)) ** 2))

        self.step_ctr += 1
        obs_arr, obs_dict = self.get_obs()

        # Check if goal has been reached
        reached_goal = self.reached_goal(torso_pos_current[0:2], self.goal)

        # Reevaluate termination condition
        done = reached_goal or self.step_ctr > 400

        if reached_goal:
            print("SUCCESS")

        # Update success rate
        if done:
            self._update_stats(reached_goal)

        ctrl_effort = np.square(ctrl).mean() * 0.01
        target_progress = (prev_dist - current_dist) * 70

        r = target_progress - ctrl_effort

        return obs_arr, r, done, obs_dict


    def step_pos(self, q):
        # Get measurements before step
        torso_pos_prev, _ = p.getBasePositionAndOrientation(self.robotId)

        # Rescale -1,1 input targets to actual joint targets
        resc_q = (2 * (q - self.joint_lim_lows) / (self.joint_lim_highs - self.joint_lim_lows)) - 1

        # Add control law
        p.setJointMotorControlArray(self.robotId, self.joint_ids, p.POSITION_CONTROL, targetPositions=q, forces=[10000] * 8)

        # Perform single step of simulation
        p.stepSimulation()

        # Get measurements after step
        torso_pos_current, _ = p.getBasePositionAndOrientation(self.robotId)

        prev_dist = np.sqrt(np.sum((np.asarray(torso_pos_prev[0:2]) - np.asarray(self.goal)) ** 2))
        current_dist = np.sqrt(np.sum((np.asarray(torso_pos_current[0:2]) - np.asarray(self.goal)) ** 2))

        self.step_ctr += 1
        obs_arr, obs_dict = self.get_obs()

        # Check if goal has been reached
        reached_goal = self.reached_goal(torso_pos_current[0:2], self.goal)

        # Reevaluate termination condition
        done = reached_goal or self.step_ctr > 400

        if reached_goal:
            print("SUCCESS")

        # Update success rate
        if done:
            self._update_stats(reached_goal)

        target_progress = (prev_dist - current_dist) * 70

        r = target_progress

        return obs_arr, r, done, obs_dict


    def reset(self):

        # Sample new target goal
        torso_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        self.goal = self._sample_goal(torso_pos[0:2])

        # Reset env variables
        self.step_ctr = 0

        # Variable positions
        obs_dict = {'torso_pos': (0, 0, 0.4),
                    'torso_quat': (0, 0, 0, 1),
                    'target_pos' : self.goal,
                    'q': np.zeros(self.n_joints),
                    'torso_vel': np.zeros(3),
                    'torso_angvel': np.zeros(3),
                    'q_vel': np.zeros(self.n_joints)}

        obs_arr = np.concatenate([v for v in obs_dict.values()]).astype(np.float32)

        # Target pos
        target_pos = (self.goal[0], self.goal[1], 0)

        # Set environment state
        p.resetBasePositionAndOrientation(self.robotId, obs_dict['torso_pos'], obs_dict['torso_quat'])
        p.resetBasePositionAndOrientation(self.targetId, target_pos, (0,0,0,1))
        p.resetBaseVelocity(self.robotId, obs_dict['torso_vel'])

        for j in self.joint_ids:
            p.resetJointState(self.robotId, j, 0, 0)

        return obs_arr, obs_dict


    def demo(self):
        self.reset()
        t1 = time.time()
        iters = 10000
        for i in range(iters):
            time.sleep(0.005)
            if i % 1000 == 0:
                print("Step: {}/{}".format(i, iters))
                self.reset()
            self.step_pos(np.random.randn(self.n_joints))
            #self.step_pos([-1] * 8)
        t2 = time.time()
        print("Time Elapsed: {}".format(t2-t1))


    def test(self, policy):
        self.reset()
        for i in range(100):
            done = False
            obs, _ = self.reset()
            cr = 0
            while not done:
                action = policy.sample_action(my_utils.to_tensor(obs, True)).detach()
                obs, r, done, od, = self.step(action[0])
                cr += r
                time.sleep(0.001)
            print("Total episode reward: {}".format(cr))


    def random_action(self):
        return np.random.randn(self.n_joints)


if __name__ == "__main__":
    ant = AntReach(True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
