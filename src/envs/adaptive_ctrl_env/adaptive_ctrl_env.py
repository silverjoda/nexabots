import numpy as np

class AdaptiveSliderEnv:
    def __init__(self):
        self.mass_variety = 1.
        self.target = 0
        self.target_change_prob = 0.05
        self.mem_dim = 2
        self.dt = 0.05
        self.max_steps = 500
        self.step_ctr = 0
        self.x = 0
        self.dx = 0
        self.obs_dim = 3 + self.mem_dim
        self.act_dim = 1 + self.mem_dim


    def reset(self):
        self.x = 0
        self.dx = 0
        self.target = np.random.randn()
        self.mass = 1. + np.random.rand() * self.mass_variety
        self.step_ctr = 0

        if self.mem_dim > 0:
            obs = np.concatenate((np.array(self.x, self.dx, self.target), np.zeros(self.mem_dim)))
        else:
            obs = np.array(self.x, self.dx, self.target)
        return obs


    def step(self, ctrl):

        if self.mem_dim > 0:
            act = ctrl[:self.mem_dim]
            mem = ctrl[-self.mem_dim:]
        else:
            act = ctrl

        prev_dist = np.square(self.x - self.target)

        a = act / self.mass
        self.dx += a * self.dt
        self.x += self.dx * self.dt

        self.step_ctr += 1
        done = self.step_ctr >= self.max_steps

        curr_dist = np.square(self.x - self.target)
        r = prev_dist - curr_dist

        if np.random.rand() < self.target_change_prob:
            self.target = np.random.randn()

        if self.mem_dim > 0:
            obs = np.concatenate((np.array(self.x, self.dx, self.target), mem))
        else:
            obs = np.array(self.x, self.dx, self.target)

        return obs, r, done, None

