import numpy as np
import cv2
import src.my_utils as my_utils
import time
import socket

class SliderEnv():
    def __init__(self, mass_std=0, damping_std=0.1, animate=False):
        # Initiation vars
        self.mass_std = mass_std
        self.decay_std = damping_std
        self.animate = animate

        # Simulator parameters
        self.dt = 0.1
        self.max_steps = 150
        self.obs_dim = 2
        self.act_dim = 1

        # Episode variables
        self.target = 0
        self.target_change_prob = 0.01
        self.step_ctr = 0
        self.x = 0
        self.dx = 0


    def step(self, ctrl):
        act = np.clip(ctrl[0], -1, 1)
        self.current_act = act

        # Step system
        a = act / self.mass
        self.dx += a * self.dt
        self.dx *= self.decay
        self.x += self.dx * self.dt

        # Render
        if self.animate:
            self.render()

        # Barrier condition
        if self.x > 4:
            self.x = 3.9
            self.dx = 0.
        if self.x < -4:
            self.x = -3.9
            self.dx = 0.

        self.step_ctr += 1
        done = (self.step_ctr >= self.max_steps)

        square_dist_from_target = np.square(self.x - self.target)
        vel_pen = 0.05 * np.square(self.dx) * (1 / (square_dist_from_target * 5 + 0.1))
        penalty = square_dist_from_target + vel_pen

        if abs(penalty) < 0.01:
            penalty = 0

        r = 1 / (penalty + 1)

        if np.random.rand() < self.target_change_prob:
            self.target = np.random.rand() * 6 - 3

        obs = np.array([self.x - self.target, self.dx])

        return obs, r, done, None


    def reset(self):
        self.x = 0.
        self.dx = 0.
        self.target = np.random.rand() * 6 - 3
        self.mass = 0.01 + np.random.rand() * self.mass_std
        self.decay = 0.9 + np.random.rand() * self.decay_std
        self.step_ctr = 0

        obs = np.array([self.x - self.target, self.dx])
        return obs


    def render(self):
        imdim = (48, 512)
        halfheight = int(imdim[0] / 2)
        halfwidth = int(imdim[1] / 2)
        img = np.zeros((imdim[0], imdim[1], 3), dtype=np.uint8)
        img[halfheight, :, :] = 255
        cv2.circle(img, (halfwidth + int(self.x * 36), halfheight), int(self.mass * 70), (255, 0, 0), -1)
        cv2.arrowedLine(img, (halfwidth + int(self.x * 36), halfheight), (halfwidth + int(self.x * 36) + int(self.current_act * 40), halfheight), (0, 0, 255), thickness=2)
        cv2.rectangle(img, (halfwidth + int(self.target * 36) - 1, halfheight - 5), (halfwidth + int(self.target * 36) + 1, halfheight + 5), (0, 255, 0), 1)
        cv2.putText(img, 'm = {0:.2f}'.format(self.mass), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, 'b = {0:.2f}'.format(self.decay), (80, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
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
                h = h_
                obs, r, done, od, = self.step(action[0].detach().numpy())
                cr += r
                total_rew += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))
        print("Total reward: {}".format(total_rew))


    def demo(self):
        for i in range(100):
            self.reset()
            for j in range(self.max_steps):
                self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.render()


if __name__ == "__main__":
    env = SliderEnv(mass_std=1, damping_std=0, animate=True)
    env.demo()