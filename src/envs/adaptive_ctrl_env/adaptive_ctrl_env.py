import numpy as np
import cv2

class AdaptiveSliderEnv:
    def __init__(self):
        self.mass_variety = 0.
        self.target = 0
        self.target_change_prob = 0.01
        self.render_prob = 0.07
        self.mem_dim = 0
        self.dt = 0.05
        self.max_steps = 400
        self.step_ctr = 0
        self.x = 0
        self.dx = 0
        self.obs_dim = 3 + self.mem_dim
        self.act_dim = 1 + self.mem_dim
        self.current_act = None

        cv2.namedWindow('image')


    def reset(self):
        self.x = 0
        self.dx = 0
        self.target = np.random.randn()
        self.mass = 1. + np.random.rand() * self.mass_variety
        self.step_ctr = 0
        self.render_episode = True if np.random.rand() < self.render_prob else False

        obs = np.concatenate((np.array([self.x, self.dx, self.target]), np.zeros(self.mem_dim)))
        return obs


    def render(self):
        if self.render_episode:
            imdim = (48, 512)
            halfheight = int(imdim[0] / 2)
            halfwidth = int(imdim[1] / 2)
            img = np.zeros((imdim[0], imdim[1], 3), dtype=np.uint8)
            img[halfheight, :, :] = 255
            cv2.circle(img, (halfwidth + self.x * 36, halfheight), int(self.mass * 5), (255, 0, 0), -1)
            cv2.arrowedLine(img, (halfwidth + self.x * 36, halfheight), (halfwidth + self.x * 36 + self.current_act * 20, halfheight), (0, 0, 255), thickness=3)
            cv2.rectangle(img, (halfwidth + int(self.target * 36) - 1, halfheight - 5), (halfwidth + int(self.target * 36) + 1, halfheight + 5), (0, 255, 0), 1)
            cv2.putText(img, 'm = {}'.format(self.mass), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('image', img)
            cv2.waitKey(10)


    def step(self, ctrl):
        self.current_act = ctrl[0]
        if self.mem_dim > 0:
            act = ctrl[:-self.mem_dim]
            mem = ctrl[-self.mem_dim:]
        else:
            act = ctrl
            mem = np.zeros(0)

        prev_dist = np.square(self.x - self.target)

        a = act / self.mass
        self.dx += a * self.dt
        self.x += self.dx * self.dt

        self.step_ctr += 1
        done = (self.step_ctr >= self.max_steps) or np.abs(self.x) > 6

        curr_dist = np.square(self.x - self.target)
        r = (prev_dist - curr_dist)[0] - np.square(act) * 0.01

        if np.random.rand() < self.target_change_prob:
            self.target = np.random.randn()

        obs = np.concatenate((np.array([self.x, self.dx, self.target]), mem))

        return obs, r[0], done, None

