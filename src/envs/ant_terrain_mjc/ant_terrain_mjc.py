import numpy as np
import mujoco_py
import src.my_utils as my_utils
import time
import os
import cv2

class AntTerrainMjc:
    def __init__(self, animate=False, sim=None, camera=False):
        if sim is not None:
            self.sim = sim
            self.model = self.sim.model
        else:
            self.modelpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/ant_terrain_mjc.xml")
            self.model = mujoco_py.load_model_from_path(self.modelpath)
            self.sim = mujoco_py.MjSim(self.model)

        self.camera = camera
        self.HF = True
        self.HF_div = 5

        if self.HF:
            self.hf_data = self.model.hfield_data
            self.hf_ncol = self.model.hfield_ncol[0]
            self.hf_nrow = self.model.hfield_nrow[0]
            self.hf_size = self.model.hfield_size[0]
            self.hf_grid = self.hf_data.reshape((self.hf_nrow, self.hf_ncol))
            self.hf_grid_aug = np.zeros((self.hf_nrow * 2, self.hf_ncol * 2))
            self.hf_grid_aug[:self.hf_nrow, :self.hf_ncol] = self.hf_grid
            self.hf_m_per_cell = float(self.hf_size[1]) / self.hf_nrow
            self.rob_dim = 0.5
            self.hf_res = int(self.rob_dim / self.hf_m_per_cell)
            self.hf_offset_x = 4
            self.hf_offset_y = 3

        self.model.opt.timestep = 0.02

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = self.q_dim + self.qvel_dim
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Environent inner parameters
        self.viewer = None
        self.step_ctr = 0

        if camera:
            self.cam_viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0)

        self.frame_list = []

        # Initial methods
        if animate:
            self.setupcam()

        self.reset()


        # TODO: CONTACT INPUTS


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
        a = qpos + qvel
        return np.asarray(a, dtype=np.float32)


    def get_obs_dict(self):
        od = {}

        # Intrinsic parameters
        for j in self.sim.model.joint_names:
            od[j + "_pos"] = self.sim.data.get_joint_qpos(j)
            od[j + "_vel"] = self.sim.data.get_joint_qvel(j)

        # Height field
        if self.HF:
            od["hf"] = self.get_local_hf(*od["root_pos"][0:2])

        if self.camera:
            # On board camera input
            cam_array = self.sim.render(camera_name="frontal", width=128, height=128)
            img = cv2.cvtColor(np.flipud(cam_array), cv2.COLOR_BGR2GRAY)
            #self.frame_list.append(img)
            od['cam'] = img

        return od


    def get_local_hf(self, x, y):
        x_coord = int((x + self.hf_offset_x) * 5)
        y_coord = int((y + self.hf_offset_y) * 5)
        return self.hf_grid_aug[y_coord - self.hf_res: y_coord + self.hf_res,
               x_coord - self.hf_res: x_coord + self.hf_res]


    def get_state(self):
        return self.sim.get_state()


    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.qvel_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()


    def render(self, human=True):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        if not human:
            return self.sim.render(camera_name=None,
                                   width=224,
                                   height=224,
                                   depth=False)
            #return viewer.read_pixels(width, height, depth=False)

        self.viewer.render()


    def step(self, ctrl):

        obs_p = self.get_obs()

        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        #print(self.sim.data.ncon) # Prints amount of current contacts
        obs_c = self.get_obs()
        x,y,z = obs_c[0:3]

        # Reevaluate termination condition
        done = self.step_ctr > 400 or z < 0.1

        ctrl_effort = np.square(ctrl).sum() * 0.03
        target_progress = (obs_c[0] - obs_p[0]) * 70

        r = target_progress - ctrl_effort

        return obs_c.astype(np.float32), r, done, self.get_obs_dict()


    def demo(self):
        self.reset()
        if self.HF:
            cv2.namedWindow("HF")

        qpos = [0] * self.q_dim
        for i in range(3000):
            #_, _, _, od = self.step(np.random.randn(self.act_dim))
            od = self.get_obs_dict()
            qpos[0] = i / 100.
            qpos[2] = 0.5
            self.set_state(qpos=qpos)
            self.sim.forward()
            self.sim.step()

            self.render()

            if self.HF:
                hf = od['hf']
                cv2.imshow("HF", np.flipud(hf))
                cv2.waitKey(1)
                pass

        # print("R")
        # cv2.namedWindow("camy")
        # for i in range(1000):
        #     cv2.imshow("camy", self.frame_list[i])
        #     cv2.waitKey(1)
        #     time.sleep(0.01)


    def test(self, policy):
        self.reset()
        for i in range(100):
            done = False
            obs, _ = self.reset()
            cr = 0
            while not done:
                action = policy(my_utils.to_tensor(obs, True)).detach()
                obs, r, done, od, = self.step(action[0])
                cr += r
                time.sleep(0.001)
                self.render()
            print("Total episode reward: {}".format(cr))


    def reset(self):

        # Reset env variables
        self.step_ctr = 0

        # Sample initial configuration
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = np.random.randn() * 0.1
        init_q[1] = np.random.randn() * 0.1
        init_q[2] = 0.80 + np.random.rand() * 0.1
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        obs = np.concatenate((init_q, init_qvel)).astype(np.float32)

        # Set environment state
        self.set_state(init_q, init_qvel)

        return obs, self.get_obs_dict()


if __name__ == "__main__":
    ant = AntTerrainMjc(animate=True, camera=False)
    print(ant.obs_dim)
    print(ant.act_dim)
    ant.demo()
