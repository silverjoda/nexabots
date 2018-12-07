import numpy as np
import pybullet as p
import pybullet_data
from collections import deque
import time

class AntReach:
    def __init__(self, GUI=True):

        # Start client
        physicsClient = p.connect(p.GUI if GUI else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set simulation parameters
        p.setGravity(0, 0, -9.81)

        # Load simulator objects
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadMJCF("ant_reach.xml")[0]

        self.joint_dict = {}
        for i in range(p.getNumJoints(self.robotId)):
            info  = p.getJointInfo(self.robotId, i)
            id, name, joint_type = info[0:3]
            joint_lim_low, joint_lim_high = info[8:10]
            if joint_type == 0:
                self.joint_dict[name] = (id, joint_lim_low, joint_lim_high)
        self.joint_ids = [j[0] for j in self.joint_dict.values()]

        self.obs_dim = 7 + 6 + p.getNumJoints(self.robotId) * 2
        self.act_dim = p.getNumJoints(self.robotId)

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
        base_q = p.getBasePositionAndOrientation(self.robotId)

        # Base velocity and angular velocity
        base_q_ = p.getBaseVelocity(self.robotId)

        # Joint states and velocities
        q = p.getJointStates(self.robotId, self.joint_ids)
        q_ = p.getJointStates(self.robotId, self.joint_ids)

        # Target position and velocity
        # Todo:

        a = None
        return np.asarray(a, dtype=np.float32)


    def get_obs_dict(self):
        od = {}
        for j in self.sim.model.joint_names:
            od[j + "_pos"] = self.sim.data.get_joint_qpos(j)
            od[j + "_vel"] = self.sim.data.get_joint_qvel(j)
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


    def quat_to_EA(self, quat):
        return quaternion.as_euler_angles(np.quaternion(quat))


    def render(self, human=True):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        if not human:
            return self.sim.render(camera_name=None,
                                   width=224,
                                   height=224,
                                   depth=False)
            #return viewer.read_pixels(width, height, depth=False)

        self.viewer.render()


    def step(self, ctrl):

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()

        print(self.sim.data.ncon)

        self.step_ctr += 1

        obs = self.get_obs()

        # Make relevant pose from observation (x,y)
        x, y, z, q1, q2, q3, q4 = self.sim.data.get_joint_qpos("root")
        pose = (x, y)

        prev_dist  = np.sqrt(np.sum((np.asarray(self.current_pose) - np.asarray(self.goal))**2))
        current_dist = np.sqrt(np.sum((np.asarray(pose) - np.asarray(self.goal))**2))

        # Check if goal has been reached
        reached_goal = self.reached_goal(pose, self.goal)

        # Reevaluate termination condition
        done = reached_goal or self.step_ctr > 400

        if reached_goal:
            print("SUCCESS")

        # Update success rate
        if done:
            self._update_stats(reached_goal)

        ctrl_effort = np.square(ctrl).sum() * 0.03
        target_progress = (prev_dist - current_dist) * 70
        target_trueness = 0

        r = target_progress - ctrl_effort

        self.current_pose = pose

        return obs, r, done, None


    def demo(self):
        self.reset()
        for i in range(1000):
            self.step(self.action_space.sample())
            self.render()


    def reset(self):

        # Reset env variables
        self.step_ctr = 0

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = np.random.randn() * 0.1
        init_q[1] = np.random.randn() * 0.1
        init_q[2] = 0.60 + np.random.rand() * 0.1
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        obs = np.concatenate((init_q, init_qvel))

        self.current_pose = init_q[0:2]
        self.goal = self._sample_goal(self.current_pose)

        # Set object position
        init_q[self.q_dim - 2:] = self.goal

        # Set environment state
        self.set_state(init_q, init_qvel)

        return obs


if __name__ == "__main__":
    ant = AntReach()
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
