import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os

class CentipedeMjc8:
    N = 8
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/Centipede{}_pd.xml".format(N))
    def __init__(self, animate=False, sim=None):
        self.obs_dim = 2
        self.act_dim = 2

        # Environent inner parameters
        self.env_size = 10
        self.step_ctr = 0
        self.max_steps = self.env_size * 2

        self.reset()


    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)

        self.viewer.render()


    def step(self, ctrl):



        return obs, r, done, _


    def reset(self):

        # Reset env variables
        self.step_ctr = 0

        return obs


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
    ant = CentipedeMjc8(animate=True)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
