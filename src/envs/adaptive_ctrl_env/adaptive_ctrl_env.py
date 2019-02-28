import numpy as np
import cv2
import src.my_utils as my_utils
import time
import socket

if socket.gethostname() != "goedel":
    import gym
    from gym import spaces
    from gym.utils import seeding

class AdaptiveSliderEnv():
    def __init__(self):
        self.mass_variety = 2.
        self.damping_variety = .1
        self.target = 0
        self.target_change_prob = 0.01
        self.render_prob = 0.0

        self.mem_dim = 0
        self.dt = 0.1
        self.max_steps = 200
        self.step_ctr = 0
        self.x = 0
        self.dx = 0
        self.mass_as_input = False
        self.obs_dim = 3 + self.mem_dim
        self.act_dim = 1 + self.mem_dim
        self.current_act = None

        if self.mass_as_input:
            self.obs_dim += 1

        if socket.gethostname() != "goedel":
            self.observation_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.obs_dim,))
            self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.act_dim,))
        else:
            self.render_prob = 0.00

        print("Adaptive slider, mass_variety: {}, damping_variety: {}, mass as input: {}, mem_dim: {}".format(self.mass_variety, self.damping_variety, self.mass_as_input, self.mem_dim))


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, ctrl):
        self.current_act = ctrl[0]
        if self.mem_dim > 0:
            act = ctrl[:-self.mem_dim]
            mem = ctrl[-self.mem_dim:]
        else:
            act = ctrl
            mem = np.zeros(0)

        #prev_dist = np.square(self.x - self.target)

        a = act[0] / self.mass
        self.dx += a * self.dt
        self.dx *= self.damping
        self.x += self.dx * self.dt

        if self.x > 4:
            self.x = 3.9
            self.dx = 0.
        if self.x < -4:
            self.x = -3.9
            self.dx = 0.

        self.step_ctr += 1
        done = (self.step_ctr >= self.max_steps) # or np.abs(self.x) > 6

        #curr_dist = np.square(self.x - self.target)

        # progress = (prev_dist - curr_dist)[0]
        # if progress >= 0:
        #     r = progress
        # else:
        #     r = progress * 2

        penalty = np.square(self.x - self.target) + np.square(self.dx)

        r = 1 / (penalty + 1)

        if np.random.rand() < self.target_change_prob:
            self.target = np.random.randn()

        if self.mass_as_input:
            obs = np.concatenate((np.array([self.x, self.dx, self.target]), [self.mass], mem))
        else:
            obs = np.concatenate((np.array([self.x, self.dx, self.target]), mem))

        return obs, r, done, None


    def reset(self):
        self.x = 0.
        self.dx = 0.
        self.target = np.random.randn()
        self.mass = 0.1 + np.random.rand() * self.mass_variety
        self.damping = 0.8 + np.random.rand() * self.damping_variety
        self.step_ctr = 0
        self.render_episode = True if np.random.rand() < self.render_prob else False
        self.prev_act = 0.

        if self.mass_as_input:
            obs = np.concatenate((np.array([self.x, self.dx, self.target]), [self.mass], np.zeros(self.mem_dim)))
        else:
            obs = np.concatenate((np.array([self.x, self.dx, self.target]), np.zeros(self.mem_dim)))
        return obs


    def render(self):
        self.x = 1
        self.target = 1
        if self.render_episode:
            imdim = (48, 512)
            halfheight = int(imdim[0] / 2)
            halfwidth = int(imdim[1] / 2)
            img = np.zeros((imdim[0], imdim[1], 3), dtype=np.uint8)
            img[halfheight, :, :] = 255
            cv2.circle(img, (halfwidth + int(self.x * 36), halfheight), int(self.mass * 5), (255, 0, 0), -1)
            cv2.arrowedLine(img, (halfwidth + int(self.x * 36), halfheight), (halfwidth + int(self.x * 36) + int(self.current_act * 20), halfheight), (0, 0, 255), thickness=3)
            cv2.rectangle(img, (halfwidth + int(self.target * 36) - 1, halfheight - 5), (halfwidth + int(self.target * 36) + 1, halfheight + 5), (0, 255, 0), 1)
            cv2.putText(img, 'm = {0:.2f}'.format(self.mass), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, 'b = {0:.2f}'.format(self.damping), (80, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('image', img)
            cv2.waitKey(10)


    def test(self, policy):
        self.render_prob = 1.0
        total_rew = 0
        for i in range(100):
            obs = self.reset()
            cr = 0
            for j in range(self.max_steps):
                action = policy(my_utils.to_tensor(obs, True)).detach()
                obs, r, done, od, = self.step(action[0].numpy())
                cr += r
                total_rew += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))
        print("Total reward: {}".format(total_rew))


    def test_recurrent(self, policy):
        total_rew = 0
        self.render_prob = 1.0
        for i in range(100):
            obs = self.reset()
            h = None
            cr = 0
            for j in range(self.max_steps):
                action, h_ = policy((my_utils.to_tensor(obs, True), h))
                #print(action[0].detach().numpy()[:])
                h = h_
                obs, r, done, od, = self.step(action[0].detach().numpy())
                cr += r
                total_rew += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))
        print("Total reward: {}".format(total_rew))

