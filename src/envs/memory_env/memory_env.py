import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
from copy import deepcopy
import cv2

class MemoryEnv():
    def __init__(self, animate=False):
        self.mem_dim = 0
        self.obs_dim = 2 + self.mem_dim
        self.act_dim = 1 + self.mem_dim

        # Environent inner parameters
        self.env_size = 10
        self.half_way = int(self.env_size / 2) + 1
        self.N_points = int(self.env_size / 2)
        self.step_ctr = 0
        self.max_steps = self.env_size * 2
        self.render_prob = 0.00

        self.reset()

        if animate:
            cv2.namedWindow('image')

        assert self.env_size % 2 == 0, "Env size must be divisible by 2"
        print("MemoryEnv with mem_dim: {}, total achievable reward: {}".format(self.mem_dim, self.env_size))


    def render(self):
        if self.render_episode:
            cv2.imshow('image', self.get_env_img())
            cv2.waitKey(200)


    def step(self, ctrl):
        self.step_ctr += 1

        if self.mem_dim > 0:
            mem = ctrl[-self.mem_dim:]
            act = ctrl[:-self.mem_dim]
        else:
            mem = np.zeros(0)
            act = ctrl

        if act > 0:
            self.current_pos[1] = 1 - self.current_pos[1]
        self.current_pos[0] += 1

        done = (self.step_ctr == (self.env_size))
        r = self.board[self.current_pos[0], self.current_pos[1]]
        obs = np.concatenate((self.get_obs(), mem))

        return obs, r, done, None


    def reset(self):
        # Reset env variables
        self.step_ctr = 0

        self.render_episode = np.random.rand() < self.render_prob

        # Set up board
        self.board = np.zeros((self.env_size + 1, 2))
        self.current_pos = [0,0]

        # Randomly set points
        for i in range(1, self.N_points + 1):
            rnd_dir = np.random.randint(0,2)
            self.board[i, rnd_dir] = 1
            self.board[i + self.N_points, rnd_dir] = 1

        return np.concatenate((self.get_obs(), np.zeros(self.mem_dim)))


    def get_obs(self):
        if self.current_pos[0] >= self.half_way - 1:
            return np.array([-1, self.current_pos[1]])
        return np.array([self.board[self.current_pos[0] + 1, self.current_pos[1]], self.current_pos[1]])


    def get_env_img(self):
        img = np.tile(np.expand_dims(self.board, 2), (1,1,3))
        img[self.current_pos[0], self.current_pos[1], 0] = 1
        return img


    def test(self, policy):
        import cv2

        self.reset()
        for i in range(100):
            done = False
            obs = self.reset()
            cr = 0
            cv2.namedWindow('image')
            while not done:
                action = policy((my_utils.to_tensor(obs, True)))
                obs, r, done, od, = self.step(action.squeeze(0).detach().numpy())
                cr += r
                time.sleep(0.4)

                cv2.imshow('image', self.get_env_img())
                if cv2.waitKey(20) & 0xFF == 27:
                    break

            print("Total episode reward: {}".format(cr))
        cv2.destroyAllWindows()


    def test_recurrent(self, policy):
        import cv2

        self.reset()
        for i in range(100):
            done = False
            obs = self.reset()
            h = None
            cr = 0
            cv2.namedWindow('image')
            while not done:
                action, h = policy((my_utils.to_tensor(obs, True), h))
                action = action[0].detach().numpy()
                obs, r, done, od, = self.step(action)
                cr += r
                time.sleep(0.4)

                cv2.imshow('image', self.get_env_img())
                if cv2.waitKey(20) & 0xFF == 27:
                    break

            print("Total episode reward: {}".format(cr))
        cv2.destroyAllWindows()


