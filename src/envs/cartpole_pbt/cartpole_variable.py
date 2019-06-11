"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import numpy as np
import pybullet as p

import numpy as np
import cv2
import src.my_utils as my_utils
import time
import socket


class CartPoleBulletEnv():
    def __init__(self, animate=False, latent_input=False, action_input=False):
        if (animate):
          p.connect(p.GUI)
        else:
          p.connect(p.DIRECT)

        self.animate = animate
        self.latent_input = latent_input
        self.action_input = action_input

        # Simulator parameters
        self.max_steps = 400
        self.obs_dim = 4 + 2 * int(self.latent_input) + int(self.action_input)
        self.act_dim = 1

        self.timeStep = 0.02

        self.dist_var = 2
        self.mass_var = 2

        print(self.dist_var, self.mass_var)


    def get_obs(self):
        x, x_dot, theta, theta_dot = p.getJointState(self.cartpole, 0)[0:2] + p.getJointState(self.cartpole, 1)[0:2]

        # Clip velocities
        x_dot = np.clip(x_dot / 3, -7, 7)
        theta_dot = np.clip(theta_dot / 3, -7, 7)

        # Change theta range to [-pi, pi]
        if theta > 0:
            if theta % (2 * np.pi) <= np.pi:
                theta = theta % (2 * np.pi)
            else:
                theta = -np.pi + theta % np.pi
        else:
            if theta % (-2 * np.pi) >= -np.pi:
                theta = theta % (-2 * np.pi)
            else:
                theta = np.pi + theta % -np.pi

        theta /= np.pi

        self.state = np.array([x, x_dot, theta, theta_dot])
        return self.state


    def render(self):
        pass


    def step(self, ctrl):
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=ctrl * 20)
        p.stepSimulation()

        self.step_ctr += 1

        # x, x_dot, theta, theta_dot
        obs = self.get_obs()
        x, x_dot, theta, theta_dot = obs

        angle_rew = 0.5 - np.abs(theta)
        cart_pen = np.square(x) * 0.05
        vel_pen = (np.square(x_dot) * 0.1 + np.square(theta_dot) * 0.5) * (1 - abs(theta))

        r = angle_rew - cart_pen - vel_pen - np.square(ctrl[0]) * 0.03
        done = self.step_ctr > self.max_steps

        self.theta_prev = theta

        if self.latent_input:
            obs = np.concatenate((obs, [self.dist, self.mass]))
        if self.action_input:
            obs = np.concatenate((obs, ctrl))

        return obs, r, done, (self.dist, self.mass)


    def reset(self):
        self.step_ctr = 0
        self.theta_prev = 1

        p.disconnect()

        if (self.animate):
          p.connect(p.GUI)
        else:
          p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)

        self.cartpole = self.create_robot()

        p.resetJointState(self.cartpole, 0, targetValue=0, targetVelocity=0)
        p.resetJointState(self.cartpole, 1, targetValue=np.pi, targetVelocity=0)
        #p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0)
        #p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0)
        #p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0)
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        obs, _, _, _ = self.step(np.zeros(self.act_dim))
        return obs


    def create_robot(self):
        self.dist = 0.5 + np.random.rand() * self.dist_var
        self.mass = 0.3 + np.random.rand() * self.mass_var

        filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cartpole_gen.urdf")

        content = """<?xml version="1.0"?>
<robot name="physics">

  <link name="slideBar">
    <visual>
      <geometry>
        <box size="30 0.05 0.05"/>
      </geometry>
      <origin xyz="0 0 0"/>
      <material name="green">
        <color rgba="0 0.8 .8 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <link name="cart">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <origin xyz="0 0 0"/>
      <material name="blue">
        <color rgba="0 0 .8 1"/>
      </material>
    </visual>

    <inertial>
      <mass value="1"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <joint name="slider_to_cart" type="prismatic">
    <axis xyz="1 0 0"/>
    <origin xyz="0.0 0.0 0.0"/>
    <parent link="slideBar"/>
    <child link="cart"/>
    <limit effort="1000.0" lower="-2" upper="2" velocity="10"/>
    <dynamics damping="0.0" friction="0.0"/>
  </joint>


  <link name="pole">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <origin xyz="0 0 {}"/>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>

    <inertial>
      <origin xyz="0 0 {}"/>
      <mass value="{}"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <joint name="cart_to_pole" type="continuous">
    <axis xyz="0 1 0"/>
    <origin xyz="0.0 0.0 0"/>
    <parent link="cart"/>
    <child link="pole"/>
    <limit effort="1000.0" lower="-2" upper="2" velocity="30"/>
    <joint_properties damping="0.0" friction="0.0"/>
  </joint>


</robot>""".format(self.dist, self.dist, self.mass)

        with open(filepath, "w") as out_file:
            out_file.write(content)

        while True:
            try:
                id = p.loadURDF(filepath)
                break
            except Exception:
                pass

        return id


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
                time.sleep(0.01)
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
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.step(np.array([-0.3]))
                time.sleep(0.01)
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.step(np.array([0.3]))
                time.sleep(0.005)
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.step(np.array([-0.3]))
                time.sleep(0.01)
            for j in range(120):
                # self.step(np.random.rand(self.act_dim) * 2 - 1)
                self.step(np.array([0.3]))
                time.sleep(0.01)


if __name__ == "__main__":
    env = CartPoleBulletEnv(animate=True)
    env.demo()