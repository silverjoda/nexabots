import numpy as np
import cv2
import os
import math
from math import exp
import noise
import time
from opensimplex import OpenSimplex

def flat(res):
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.zeros((M, N), dtype=np.float32)

    # Walls
    mat[0, :] = 1.
    mat[-1, :] = 1.
    mat[:, 0] = 1.
    mat[:, -1] = 1.

    return mat, {"height" : 0.8}


def corridor(res, cw=8):
    # Make even dimensions
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.ones((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    mat[M_2 - cw: M_2 + cw, : ] = 0.4

    # Walls
    mat[:, 0] = 1.
    mat[:, -1] = 1.

    # Multiply to full image resolution
    mat *= 255

    return mat, {"height": 0.3}


def corridor_holes(res, cw=8):
    # Make even dimensions
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.zeros((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    # Amount of 'tiles'
    Mt = 2
    Nt = 25

    floor_height = 0.6

    # Makes tiles array
    p_kill = 0.5
    tiles_array = np.ones((2, Nt)) * floor_height
    for i in range(Nt):
        if np.random.rand() < p_kill:
            tiles_array[np.random.randint(0,2), i] = 0

    # Translate tiles array to real heightmap
    for i in range(Nt):
        mat[M_2 - cw: M_2, i * cw: i * cw + cw] = tiles_array[0, i]
        mat[M_2:M_2 + cw:, i * cw: i * cw + cw] = tiles_array[1, i]

    # Flat starting pos
    mat[M_2 - cw: M_2 + cw, : 4 * cw] = floor_height

    # Walls
    mat[0, 0] = 1.
    # mat[:, 0] = 1.
    # mat[:, -1] = 1.

    # Multiply to full image resolution
    mat *= 255

    return mat, {"height" : 0.8}


def tiles(res):
    cw = 10
    # Make even dimensions
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.ones((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    # Amount of 'tiles'
    Mt = 2
    Nt = 20

    # Tile height
    t_height = 0.3

    # Makes tiles array
    tiles_array = np.ones((Mt, Nt)) * 0.4
    for i in range(Nt):
        tiles_array[0, i] = np.random.rand() * t_height
        tiles_array[1, i] = np.random.rand() * t_height

    # Translate tiles array to real heightmap
    for i in range(Nt):
        mat[M_2 - cw: M_2, i * cw: i * cw + cw] = tiles_array[0, i]
        mat[M_2:M_2 + cw:, i * cw: i * cw + cw] = tiles_array[1, i]



    # Walls
    mat[:, 0] = 1.
    mat[:, -1] = 1.

    # Multiply to full image resolution
    mat *= 255

    return mat, {"height" : 0.5}


def triangles(res):


    cw = 10
    # Make even dimensions
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.zeros((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    # Amount of 'tiles'
    Mt = 2
    Nt = 20
    obstacle_height = 0.3

    grad_mat = np.linspace(0, 1, cw)[:, np.newaxis].repeat(cw, 1)
    template_1 = np.ones((cw, cw)) * grad_mat * grad_mat.T * obstacle_height
    template_2 = np.ones((cw, cw)) * grad_mat * obstacle_height

    for i in range(Nt):
        if np.random.choice([True, False]):
            mat[M_2 - cw: M_2, i * cw: i * cw + cw] = np.rot90(template_1, np.random.randint(0,4))
        else:
            mat[M_2 - cw: M_2, i * cw: i * cw + cw] = np.rot90(template_2, np.random.randint(0,4))

        if np.random.choice([True, False]):
            mat[M_2:M_2 + cw:, i * cw: i * cw + cw] = np.rot90(template_1, np.random.randint(0,4))
        else:
            mat[M_2:M_2 + cw:, i * cw: i * cw + cw] = np.rot90(template_2, np.random.randint(0,4))

    # Walls
    mat[0, 0] = 1.
    #mat[:, 0] = 1.
    #mat[:, -1] = 1.

    # Multiply to full image resolution
    mat *= 255

    return mat, {"height" : 0.5, "friction":0.2}


def domes(res):
    cw = 10
    # Make even dimensions
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.ones((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    # Amount of 'tiles'
    Mt = 2
    Nt = 20
    obstacle_height = 0.3

    grad_mat = np.linspace(-1, 1, cw)[:, np.newaxis].repeat(cw, 1)
    dome_mat = np.ones((cw, cw))
    dome_mat = dome_mat * grad_mat * np.rot90(grad_mat, 1)
    dome_mat = (1 - np.power(dome_mat, 2)) * obstacle_height

    mat[M_2 - cw:M_2 + cw, :] = 0

    for i in range(Nt):
        if i > 1 and i < Nt - 1:
            rnd_x, rnd_y = np.random.randint(-2,3, size=(2))
        else:
            rnd_x, rnd_y = 0, 0

        mat[M_2 + rnd_x:M_2 + rnd_x + cw:, i * cw + rnd_y: i * cw + rnd_y + cw] = dome_mat

        if i > 1 and i < Nt - 1:
            rnd_x, rnd_y = np.random.randint(-2,3, size=(2))
        else:
            rnd_x, rnd_y = 0, 0

        mat[M_2 + rnd_x - cw:M_2 + rnd_x:, i * cw + rnd_y: i * cw + rnd_y + cw] = dome_mat

    # Walls
    mat[:, 0] = 1.
    mat[:, -1] = 1.
    mat[0:M_2 - cw, :] = 1
    mat[M_2 + cw:, :] = 1

    # Multiply to full image resolution
    mat *= 255

    return mat, {"height" : 0.5}


def stairs(res):
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.ones((M, N), dtype=np.float32) * 0
    M_2 = math.ceil(M / 2)

    lim_asc = 1, 6
    lim_desc = 1, 6
    lim_flat = 1, 4

    N_steps = 20

    # Generate step sequence
    steps = []
    curr_height = 0
    for i in range(N_steps):
        c = np.random.choice(("u", "d", "f"), p = [0.4, 0.4, 0.2])
        seq = None
        if c == "u":
            seq = [curr_height + h for h in range(np.random.randint(lim_asc[0], lim_asc[1]))]
            curr_height += len(seq)
        if c == "d":
            seq = [max(curr_height - h, 0) for h in range(np.random.randint(lim_desc[0], lim_desc[1]))]
            curr_height = max(0, curr_height - len(seq))
        if c == "f":
            seq = [curr_height for _ in range(np.random.randint(lim_flat[0], lim_flat[1]))]
        steps.extend(seq)

    # Step dimensions
    step_len, step_height, step_width = 4, 16, 10

    # Buffer steps sequence initially with zeros
    steps.reverse()
    steps.extend([0] * 4)
    steps.reverse()

    # Fill in height map
    for i, s in enumerate(steps):
        mat[M_2 - step_width:M_2 + step_width, i * step_len: (i + 1) * step_len] = s * step_height

    # Heightmap normalization
    mat[0,0] = 255

    # Fill
    # mat[:M_2 - step_width] = 255
    # mat[M_2 + step_width :] = 255
    # mat[:, 0] = 255
    # mat[:, -1] = 255

    return mat, {"height" : 2.0}


def pipe(res, radius=8):
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.ones((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    pipe_mat = np.linspace(-radius, radius, radius * 2).repeat(N).reshape(radius * 2, N)
    mat[M_2 - radius: M_2 + radius, :] = 1 - np.sqrt(radius**2 - np.power(pipe_mat, 2)) / radius

    # Walls
    mat[:, 0] = 1.
    mat[:, -1] = 1.
    mat[0, :] = 1.
    mat[-1, :] = 1.

    mat *= 255

    return mat, {"height" : 0.5}


def slant(res):
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.zeros((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)
    W_2 = 15

    c_p = [30, 70, 120, 200]
    flat_height = 0.4

    mat[M_2 - W_2: M_2 + W_2, :c_p[0]] = np.linspace(flat_height, flat_height, W_2 * 2).repeat(c_p[0]).reshape(W_2 * 2, c_p[0])
    for i in range(len(c_p) - 1):
        d = np.random.choice(["f", "l", "r"])
        if d == "f":
            mat[M_2 - W_2: M_2 + W_2, c_p[i]: c_p[i+1]] = np.linspace(flat_height, flat_height, W_2 * 2).repeat(c_p[i+1] - c_p[i]).reshape(W_2 * 2, c_p[i+1] - c_p[i])
        if d == "l":
            mat[M_2 - W_2: M_2 + W_2, c_p[i]: c_p[i+1]] = np.linspace(1, 0, W_2 * 2).repeat(c_p[i+1] - c_p[i]).reshape(W_2 * 2, c_p[i+1] - c_p[i])
        if d == "r":
            mat[M_2 - W_2: M_2 + W_2, c_p[i]: c_p[i+1]] = np.linspace(0, 1, W_2 * 2).repeat(c_p[i+1] - c_p[i]).reshape(W_2 * 2, c_p[i+1] - c_p[i])

    s_d = 5
    for i in range(len(c_p) - 1):
        tmp = np.zeros((W_2 * 2, s_d * 2))
        for j in range(-s_d, s_d):
            tmp[:, j + s_d] = np.mean(mat[M_2 - W_2: M_2 + W_2, c_p[i] + j - s_d:c_p[i] + j + s_d], axis=1, keepdims=False)
        mat[M_2 - W_2: M_2 + W_2, c_p[i] - s_d:c_p[i] + s_d] = tmp

    # Walls
    mat[:, 0] = 1.
    mat[:, -1] = 1.
    mat[0, :] = 1.
    mat[-1, :] = 1.

    mat *= 255

    return mat, {"height" : 0.5}


def perlin(res):
    oSim = OpenSimplex(seed=int(time.time()))

    height = 200

    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.zeros((M, N))

    scale_x = np.random.randint(30, 100)
    scale_y = np.random.randint(30, 100)
    octaves = 4 # np.random.randint(1, 5)
    persistence = np.random.rand() * 0.3 + 0.3
    lacunarity = np.random.rand() + 1.5

    for i in range(M):
        for j in range(N):
            for o in range(octaves):
                sx = scale_x * (1 / (lacunarity ** o))
                sy = scale_y * (1 / (lacunarity ** o))
                amp = persistence ** o
                mat[i][j] += oSim.noise2d(i / sx, j / sy) * amp

    wmin, wmax = mat.min(), mat.max()
    mat = (mat - wmin) / (wmax - wmin) * height

    if np.random.rand() < 0.3:
        num = np.random.randint(50, 120)
        mat = np.clip(mat, num, 200)
    if np.random.rand() < 0.3:
        num = np.random.randint(120, 200)
        mat = np.clip(mat, 0, num)

    # Walls
    mat[0, 0] = 255.
    # mat[0, :] = 255.
    # mat[-1, :] = 255.
    # mat[:, 0] = 255.
    # mat[:, -1] = 255.

    return mat, {"height" : 1.2}


def corridor_various_width(res):
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.ones((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    min_width, max_width, min_length, max_length = (5, 8, 6, 14)

    c_m, c_n = (M_2, 16)

    def addbox(m, n, size_m, size_n, mat):
        mat[m - size_m: m + size_m, n - size_n: n + size_n] = 0

    # Add first box
    addbox(c_m, c_n, max_width, max_length, mat)
    c_n += max_length

    while True:
        width = np.random.randint(min_width, max_width)
        length = np.random.randint(min_length, max_length)

        c_n += length

        if c_n > N: break

        # Add to wall_list while removing overlapping walls
        addbox(c_m, c_n, width, length, mat)

        c_n += length


    return mat, {"height" : 0.3}


def pipe_variable_rad(res, min_rad=6, max_rad=12):
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.ones((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    # Initial position
    init_l = 20
    diff_rad = (max_rad - min_rad) / 2
    mid_rad = max_rad * 0.5 + min_rad * 0.5

    def rbf_kernel(x1, x2, variance=1):
        return exp(-1 * ((x1 - x2) ** 2) / (2 * variance))


    def gram_matrix(xs):
        return [[rbf_kernel(x1, x2) for x2 in xs] for x1 in xs]

    xs = np.linspace(0, 20, N)
    mean = [0 for _ in xs]
    gram = gram_matrix(xs)
    ys = np.random.multivariate_normal(mean, gram)

    for i in range(0, N):
        #rad = np.clip(rad + np.random.randint(-1, 2), min_rad, max_rad)
        rad = int(np.clip(ys[i] * 2, -diff_rad, diff_rad)  + mid_rad)
        #print(noise.pnoise1(float(prx + i)))
        pipe_mat = np.linspace(-rad, rad, rad * 2)
        mat[M_2 - rad: M_2 + rad, i] = 1 - np.sqrt(rad ** 2 - np.power(pipe_mat, 2)) / rad

    # Walls
    mat[:, 0] = 1.
    mat[:, -1] = 1.
    mat[0, :] = 1.
    mat[-1, :] = 1.

    mat *= 255

    return mat, {"height" : 0.3}


def corridor_turns(res):
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.ones((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    N_junctions = 8
    box_size = 20
    c_m, c_n = (M_2, math.ceil(box_size / 2))

    def addbox(m, n, size, mat):
        hs = math.ceil(size / 2)
        mat[m - hs: m + hs, n - hs: n + hs] = 0

    # Add first box
    addbox(c_m, c_n, box_size, mat)

    while True:
        d = np.random.choice(["N", "W", "E"], p=[0.5, 0.25, 0.25])
        if d == "N":
            # Move
            c_n += box_size
        if d == "W":
            # Move
            c_m -= box_size
        if d == "E":
            # Move
            c_m += box_size

        if c_m > M - box_size: c_m = M - box_size
        if c_m < box_size: c_m = box_size

        if c_n > N:
            break

        # Add to wall_list while removing overlapping walls
        addbox(c_m, c_n, box_size, mat)

    # Add walls
    mat[:, 0] = 1.
    mat[:, -1] = 1.

    return mat, {"height" : 0.3}


def pillars_random(res):
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.zeros((M, N), dtype=np.float32)

    block_size = 6

    def addbox(m, n, size, mat):
        hs = math.ceil(size / 2)
        mat[m - hs: m + hs, n - hs: n + hs] = 1

    # Add blocks
    for i in range(40):
            addbox(np.random.randint(block_size, M - block_size),
                   np.random.randint(block_size + 13, N - block_size),
                   block_size, mat)

    # Walls
    mat[:, 0] = 1.
    mat[:, -1] = 1.
    mat[0, :] = 1.
    mat[-1, :] = 1.

    return mat, {"height" : 0.3}


def pillars_pseudorandom(res):
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.zeros((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    block_size = 20

    M_tiles = math.ceil(M / block_size)
    N_tiles = math.ceil(N / block_size)

    def addbox(m, n, size, mat):
        hs = math.ceil(size / 2)
        mat[m - hs: m + hs, n - hs: n + hs] = 1

    # Add blocks
    for m in range(M_tiles - 1):
        for n in range(N_tiles - 1):
            addbox(m * block_size + block_size + np.random.randint(math.ceil(-block_size / 2), math.ceil(block_size / 2)),
                   n * block_size + block_size + np.random.randint(math.ceil(-block_size / 2), math.ceil(block_size / 2)),
                   block_size / 4, mat)

    # Clear initial starting position
    mat[M_2 - block_size:M_2 + block_size, :14] = 0.

    # Walls
    mat[:, 0] = 1.
    mat[:, -1] = 1.
    mat[0, :] = 1.
    mat[-1, :] = 1.

    return mat, {"height" : 0.3}


if __name__ == "__main__":
    pass