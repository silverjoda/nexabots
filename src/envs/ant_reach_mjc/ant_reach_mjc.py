import time
import os
import numpy as np
import mujoco_py
from collections import deque
import src.my_utils as my_utils

class AntReachMjc:
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ant_reach.xml")
    def __init__(self, animate=False, sim=None):
        if sim is not None:
            self.sim = sim
            self.model = self.sim.model
        else:
            self.modelpath = AntReachMjc.MODELPATH
            self.model = mujoco_py.load_model_from_path(self.modelpath)
            self.sim = mujoco_py.MjSim(self.model)

        self.model.opt.timestep = 0.02
        self.animate = animate

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = self.q_dim + self.qvel_dim
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Environent inner parameters
        self.success_queue = deque(maxlen=100)
        self.viewer = None
        self.goal = None
        self.current_pose = None
        self.success_rate = 0
        self.step_ctr = 0

        # Initial methods
        self.reset()

        if self.animate:
            self.setupcam()


    def setupcam(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * 1.3
        self.viewer.cam.lookat[0] = -0.1
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -20


    def get_obs(self):
        qpos = self.sim.get_state().qpos.tolist()
        qvel = self.sim.get_state().qvel.tolist()
        a = qpos + qvel
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


    def render(self):
        if not self.animate:
            return
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()


    def step(self, ctrl):

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()

        #print(self.sim.data.ncon)

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
        done = reached_goal or self.step_ctr > 300

        #if reached_goal:
        #    print("SUCCESS")

        # Update success rate
        if done:
            self._update_stats(reached_goal)

        ctrl_effort = np.square(ctrl).sum() * 0.05
        target_progress = (prev_dist - current_dist) * 70

        r = target_progress - ctrl_effort

        self.current_pose = pose

        return obs, r, done, None


    def demo(self):
        self.reset()
        for i in range(1000):
            self.step(np.random.randn(self.act_dim))
            self.render()


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
                self.render()
            print("Total episode reward: {}".format(cr))


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

        return obs, None


if __name__ == "__main__":
    ant = AntReachMjc(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
