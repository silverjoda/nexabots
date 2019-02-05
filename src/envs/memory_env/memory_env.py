import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os

class MemoryEnv:
    def __init__(self, animate=False):
        self.obs_dim = 2
        self.act_dim = 2

        # Environent inner parameters
        self.env_size = 10
        self.N_points = self.env_size / 2
        self.step_ctr = 0
        self.max_steps = self.env_size * 2

        self.reset()


    def render(self):
        pass


    def step(self, ctrl):
        if self.stage == 0:
            pass
        else:
            pass

        obs = self.get_obs()

        r = 0
        done = False

        return obs, r, done, None


    def reset(self):

        # Reset env variables
        self.step_ctr = 0

        # Set up board
        self.board = np.zeros((self.env_size, 2))
        self.current_pos = (0,0)
        self.stage = 0

        # Randomly set points
        pts = np.random.choice(self.env_size, self.N_points, replace=False)
        for p in pts:
            self.board[p, np.random.randint(0,2)] = 1

        return self.get_obs()


    def get_obs(self):
        if self.stage == 0:
            return [self.board[self.current_pos[0] + 1], self.stage]
        else:
            return [self.board[self.current_pos[0] - 1], self.stage]


    def demo(self):
        self.reset()
        for i in range(1000):
            self.step(np.random.randn(self.act_dim))
            self.render()


    def test(self, policy):
        self.reset()
        for i in range(100):
            done = False
            obs = self.reset()
            cr = 0
            for i in range(1000):
                action = policy(my_utils.to_tensor(obs, True))[0].detach()
                obs, r, done, od, = self.step(action[0])
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))


    def test_recurrent(self, policy):
        self.reset()
        for i in range(100):
            done = False
            obs = self.reset()
            h = policy.init_hidden()
            cr = 0
            while not done:
                action, h_ = policy((my_utils.to_tensor(obs, True), h))
                h = h_
                obs, r, done, od, = self.step(action[0].detach())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))





if __name__ == "__main__":
    ant = MemoryEnv(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
