import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
from copy import deepcopy

class MemoryEnv:
    def __init__(self, animate=False):
        self.obs_dim = 3
        self.act_dim = 1

        # Environent inner parameters
        self.env_size = 10
        self.N_points = int(self.env_size / 2)
        self.step_ctr = 0
        self.max_steps = self.env_size * 2

        self.reset()


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
        if self.stage == 0:
            return [self.board[self.current_pos[0] + 1, 0],
                    self.board[self.current_pos[0] + 1, 1],
                    self.current_pos[1]]
        else:
            return [-1, -1, self.current_pos[1]]


    def get_env_img(self):
        img = np.tile(np.expand_dims(self.board, 2), (1,1,3))
        img[self.current_pos[0], self.current_pos[1], 0] = 1
        return img


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
                action_dist, h = policy((my_utils.to_tensor(obs, True), h))
                action = np.argmax(action[0].detach().numpy())
                obs, r, done, od, = self.step(action)
                cr += r
                time.sleep(0.001)

                cv2.imshow('image', self.get_env_img())
                if cv2.waitKey(20) & 0xFF == 27:
                    break

            print("Total episode reward: {}".format(cr))
        cv2.destroyAllWindows()


