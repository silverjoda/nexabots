import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
from math import sqrt, acos, fabs
import random
import string


class Cartpole:
    def __init__(self, animate=False):
        self.modelpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/tst.xml")
        self.max_steps = 200
        self.mem_dim = 0
        self.cumulative_environment_reward = None
        self.len_as_input = False
        self.fixed = False
        print("Cartpole swingup, fo: {}, fixed: {}".format(self.len_as_input, self.fixed))

        self.make_env()

        self.obs_dim = 4 if not self.len_as_input else 5
        self.act_dim = 1

        # Environent inner parameters
        self.viewer = None

        # Reset env variables
        self.step_ctr = 0
        self.episodes = 0

        self.reset()


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
        a = [qpos[0], qpos[1], qvel[0] / 10, qvel[1] / 10]
        if self.len_as_input:
            a += [self.rnd_len]
        return np.asarray(a, dtype=np.float32)


    def get_state(self):
        return self.sim.get_state()


    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.q_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()


    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()


    def step(self, ctrl):
        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        obs = self.get_obs()
        r = 1 + np.cos(obs[1] + np.pi) \
            - np.square(obs[0]) * 0.5 \
            - np.square(ctrl[0]) * 0.01 \
            - 0.1 * np.square(obs[3]) \
            - 0.1 * np.square(obs[2])

        done = self.step_ctr > self.max_steps #or np.abs(obs[3]) > 3 or np.abs(obs[0]) > 0.97

        return obs, r, done, None


    def reset(self):
        self.step_ctr = 0

        if not self.fixed:
            self.make_env()

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_qvel = np.zeros(self.qvel_dim, dtype=np.float32)

        #init_q[1] = np.pi

        # Set environment state
        self.set_state(init_q, init_qvel)
        obs = self.get_obs()

        return obs


    def make_env(self):
        self.viewer = None
        if self.fixed:
            self.rnd_len = -0.6
        else:
            self.rnd_len = -(0.4 + np.random.rand() * 0.9)

        # Generate new xml
        self.generate_new(self.rnd_len)

        while True:
            try:
                self.model = mujoco_py.load_model_from_path(self.modelpath)
                break
            except Exception:
                "Retrying xml"

        self.sim = mujoco_py.MjSim(self.model)

        self.model.opt.timestep = 0.02

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

    def generate_new(self, rnd_len):
        self.test_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/tst.xml")
        content = """<mujoco model="inverted pendulum">
        <compiler inertiafromgeom="true"/>
        <default>
            <joint armature="0" damping="0.5" limited="true"/>
            <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
            <tendon/>
            <motor ctrlrange="-1 1"/>
        </default>
        <option gravity="0 0 -9.81" integrator="RK4" timestep="0.02"/>
        <size nstack="3000"/>
        <worldbody>
            <!--geom name="ground" type="plane" pos="0 0 0" /-->
            <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule"/>
            <body name="cart" pos="0 0 0">
                <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
                <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
                <body name="pole" pos="0 0 0">
                    <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-1000 1000" type="hinge"/>
                    <geom fromto="0 0 0 0.001 0 {}" name="cpole" rgba="0 0.7 0.7 1" size="0.049 0.3" type="capsule"/>
                    <!--                 <body name="pole2" pos="0.001 0 0.6"><joint name="hinge2" type="hinge" pos="0 0 0" axis="0 1 0"/><geom name="cpole2" type="capsule" fromto="0 0 0 0 0 0.6" size="0.05 0.3" rgba="0.7 0 0.7 1"/><site name="tip2" pos="0 0 .6"/></body>-->
                </body>
            </body>
        </worldbody>
        <actuator>
            <motor gear="300" joint="slider" name="slide"/>
        </actuator>
    </mujoco>""".format(rnd_len)

        with open(self.test_path, "w") as out_file:
            out_file.write(content)


    def demo(self):
        self.reset()
        for i in range(10000):
            obs, r, _, _ = self.step(np.random.randn(self.act_dim))
            self.render()


    def test(self, policy):
        for i in range(100):
            obs = self.reset()
            cr = 0
            for j in range(self.max_steps):
                action = policy(my_utils.to_tensor(obs, True)).detach()
                obs, r, done, od, = self.step(action[0].numpy())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))


    def test_recurrent(self, policy):
        self.reset()
        for i in range(100):
            obs = self.reset()
            h = None
            cr = 0
            for j in range(self.max_steps):
                action, h = policy((my_utils.to_tensor(obs, True).unsqueeze(0), h))
                obs, r, done, od, = self.step(action[0, 0].detach().numpy())
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))


if __name__ == "__main__":
    cp = Cartpole(animate=True)
    print(cp.obs_dim)
    print(cp.act_dim)
    cp.demo()