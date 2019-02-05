import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os

class MemoryEnv:
    def __init__(self, animate=False):
        self.obs_dim = 4
        self.act_dim = 1

        # Environent inner parameters
        self.env_size = 10
        self.N_points = int(self.env_size / 2)
        self.step_ctr = 0
        self.max_steps = self.env_size * 2

        self.reset()


    def render(self):
        pass


    def step(self, ctrl):
        done = False

        if ctrl == 1:
            self.current_pos[1] = 1 - self.current_pos[1]

        if self.stage == 0:
            # Didn't read end yet
            if self.current_pos[0] < self.env_size - 1:
                self.current_pos[0] += 1
            else:
                self.stage = 1
        else:
            if self.current_pos[0] > 0:
                self.current_pos[0] -= 1
            else:
                done = True

        r = self.board[self.current_pos[0]]
        obs = self.get_obs()

        return obs, r, done, None


    def reset(self):

        # Reset env variables
        self.step_ctr = 0

        # Set up board
        self.board = np.zeros((self.env_size, 2))
        self.current_pos = [0,0]
        self.stage = 0

        # Randomly set points
        pts = np.random.choice(self.env_size, self.N_points, replace=False)
        for p in pts:
            self.board[p, np.random.randint(0,2)] = 1

        return self.get_obs()


    def get_obs(self):
        inc = -1
        if self.stage == 0:
            inc = 1

        return [self.board[self.current_pos[0] + inc, 0],
                self.board[self.current_pos[0] + inc, 1],
                self.current_pos[1],
                self.stage]


    def test_recurrent(self, policy):
        self.reset()
        for i in range(100):
            done = False
            obs = self.reset()
            h = None
            cr = 0
            while not done:
                action, h_ = policy((my_utils.to_tensor(obs, True), h))
                h = h_
                obs, r, done, od, = self.step(action[0].detach())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))



