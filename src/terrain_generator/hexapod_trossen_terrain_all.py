import numpy as np
import cv2
import os
import math
from opensimplex import OpenSimplex
import time

def generate_hybrid_env(path, env_list, n_envs, env_len, env_width, replace=False, walls=False, blending_bound=3):
    '''

    :param path: Path to saved png file containing the heightmap
    :param env_list: List of terrains that we want to sampele a compound env from. These can be ['flat', 'tiles', 'slants', 'pipe', 'perlin', 'stairs']
    :param n_envs: Amount of segments in compound env
    :param env_len: Length of env
    :param replace: If true then the terrains in the compound env will be sampled with replacement
    :param walls: If true then env will have walls which prevent the robot from falling off the edge
    :param blending_bound: Smoothing factor when blending between terrains
    :return: Returns sampled env list, sizes of the environments and scaled indeces of transitions
    '''
    # Sample envs from env list
    envs = np.random.choice(env_list, n_envs, replace=replace)

    if n_envs == 1:
        size_list = [env_len]
        scaled_indeces_list = [0]
    else:
        size_list = []
        raw_indeces = np.linspace(0, 1, n_envs + 1)[1:-1]
        current_idx = 0
        scaled_indeces_list = []
        for idx in raw_indeces:
            idx_scaled = int(env_len * idx) + np.random.randint(0, int(env_len / 6)) - int(env_len / 12)
            scaled_indeces_list.append(idx_scaled)
            size_list.append(idx_scaled - current_idx)
            current_idx = idx_scaled
        size_list.append(env_len - sum(size_list))

    maplist = []
    current_height = 0
    for m, s in zip(envs, size_list):
        hm, current_height = generate_heightmap(m, s, env_width, current_height)
        maplist.append(hm)
    total_hm = np.concatenate(maplist, 1)
    heighest_point = np.max(total_hm)
    height_SF = max(heighest_point / 255., 1)
    total_hm /= height_SF
    total_hm = np.clip(total_hm, 0, 255).astype(np.uint8)

    #Smoothen transitions
    if n_envs > 1:
        for s in scaled_indeces_list:
            total_hm_copy = np.array(total_hm)
            for i in range(s - blending_bound, s + blending_bound):
                total_hm_copy[:, i] = np.mean(total_hm[:, i - blending_bound:i + blending_bound], axis=1)
            total_hm = total_hm_copy

    if walls:
        total_hm[0, :] = 255
        total_hm[:, 0] = 255
        total_hm[-1, :] = 255
        total_hm[:, -1] = 255
    else:
        total_hm[0, 0] = 255

    cv2.imwrite(path, total_hm)

    return envs, size_list, scaled_indeces_list


def generate_heightmap(env_name, env_length, env_width, current_height):
    if env_name == "flat":
        hm = np.ones((env_width, env_length)) * current_height

    if env_name == "tiles":
        sf = 3
        hm = np.random.randint(0, 55,
                               size=(env_width // sf, env_length // sf)).repeat(sf, axis=0).repeat(sf, axis=1)
        hm_pad = np.zeros((env_width, env_length))
        hm_pad[:hm.shape[0], :hm.shape[1]] = hm
        hm = hm_pad + current_height

    if env_name == "pipe":
        pipe_form = np.square(np.linspace(-1.2, 1.2, env_width))
        pipe_form = np.clip(pipe_form, 0, 1)
        hm = 255 * np.ones((env_width, env_length)) * pipe_form[np.newaxis, :].T
        hm += current_height


    if env_name == "stairs":
        hm = np.ones((env_width, env_length)) * current_height
        stair_height = 45
        stair_width = 4

        initial_offset = 0
        n_steps = math.floor(env_length / stair_width) - 1

        for i in range(n_steps):
            hm[:, initial_offset + i * stair_width: initial_offset  + i * stair_width + stair_width] = current_height
            current_height += stair_height

        hm[:, n_steps * stair_width:] = current_height


    if env_name == "slants":
        cw = 10
        # Make even dimensions
        M = math.ceil(env_width)
        N = math.ceil(env_length)
        hm = np.zeros((M, N), dtype=np.float32)
        M_2 = math.ceil(M / 2)

        # Amount of 'tiles'
        Mt = 2
        Nt = int(env_length / 10.)
        obstacle_height = 50
        grad_mat = np.linspace(0, 1, cw)[:, np.newaxis].repeat(cw, 1)
        template_1 = np.ones((cw, cw)) * grad_mat * grad_mat.T * obstacle_height
        template_2 = np.ones((cw, cw)) * grad_mat * obstacle_height

        for i in range(Nt):
            if np.random.choice([True, False]):
                hm[M_2 - cw: M_2, i * cw: i * cw + cw] = np.rot90(template_1, np.random.randint(0, 4))
            else:
                hm[M_2 - cw: M_2, i * cw: i * cw + cw] = np.rot90(template_2, np.random.randint(0, 4))

            if np.random.choice([True, False]):
                hm[M_2:M_2 + cw:, i * cw: i * cw + cw] = np.rot90(template_1, np.random.randint(0, 4))
            else:
                hm[M_2:M_2 + cw:, i * cw: i * cw + cw] = np.rot90(template_2, np.random.randint(0, 4))

        hm += current_height


    if env_name == "perlin":
        oSim = OpenSimplex(seed=int(time.time()))

        height = 120

        M = math.ceil(env_width)
        N = math.ceil(env_length)
        hm = np.zeros((M, N), dtype=np.float32)

        scale_x = 20
        scale_y = 20
        octaves = 4  # np.random.randint(1, 5)
        persistence = 1
        lacunarity = 2

        for i in range(M):
            for j in range(N):
                for o in range(octaves):
                    sx = scale_x * (1 / (lacunarity ** o))
                    sy = scale_y * (1 / (lacunarity ** o))
                    amp = persistence ** o
                    hm[i][j] += oSim.noise2d(i / sx, j / sy) * amp

        wmin, wmax = hm.min(), hm.max()
        hm = (hm - wmin) / (wmax - wmin) * height
        hm += current_height


    return hm, current_height


if __name__ == "__main__":
    generate_hybrid_env("test.png", ["stairs", "pipe", "tiles"], 3, 200, 20, replace=False, walls=False, blending_bound=1)