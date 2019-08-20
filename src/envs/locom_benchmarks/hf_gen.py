import numpy as np
import cv2
import os
import math


def hm_flat(res):
    M = int(res * 100)
    N = int(res * 100)

    return np.zeros((M, N), dtype=np.float32)


def hm_corridor(res):
    # Make even dimensions
    M = math.ceil(res * 10) * 2
    N = math.ceil(res * 100) * 2
    mat = np.zeros((M, N), dtype=np.float32)

    # Walls
    mat[0, 0:] = 1.
    mat[:, 0:] = 1.
    mat[0:, 0] = 1.
    mat[0:, :] = 1.
    return mat


def hm_corridor_holes(res):
    # Make even dimensions
    M = math.ceil(res * 10) * 2
    N = math.ceil(res * 100) * 2
    mat = np.zeros((M, N), dtype=np.float32)

    # Amount of 'tiles'
    Mt = 2
    Nt = 20
    Mt_res = M / Mt
    Nt_res = N / Nt

    # Makes tiles array
    p_kill = 0.1
    tiles_array = np.random.rand(Mt, Nt)
    tiles_array[tiles_array < p_kill] = 0
    tiles_array[tiles_array > 0] = 1

    # Translate tiles array to real heightmap
    for i in range(tiles_array.shape[1]):
        mat[0:Mt_res, i * Nt_res: i * Nt_res + Nt_res] = tiles_array[0, i]
        mat[Mt_res:, i * Nt_res: i * Nt_res + Nt_res] = tiles_array[1, i]

    # Walls
    mat[0, 0:] = 1.
    mat[:, 0:] = 1.
    mat[0:, 0] = 1.
    mat[0:, :] = 1.

    return mat


def hm_corridor_turns(res):
    # Make even dimensions
    M = math.ceil(res * 10) * 2
    N = math.ceil(res * 100) * 2
    mat = np.zeros((M, N), dtype=np.float32)

    N_junctions = 6
    box_size = 30
    c_x, c_y = (box_size / 2, 0)
    wall_set = []

    def getbox(x, y, size):
        halfsize = size / 2
        return (x - halfsize, x + halfsize, y - halfsize), \
               (x - halfsize, x + halfsize, y + halfsize), \
               (y - halfsize, y + halfsize, x - halfsize), \
               (y - halfsize, y + halfsize, x + halfsize)

    # Add first box
    [wall_set.add(w) for w in getbox(c_x, c_y, box_size)]

    for i in range(N_junctions):
        d = np.random.choice(["N", "W", "E"])
        if d == "N":
            # Move
            c_x += box_size
        if d == "W":
            # Move
            c_y -= box_size
        if d == "E":
            # Move
            c_y += box_size

        # Add to wall_list while removing overlapping walls
        [wall_set.add(w) for w in getbox(c_x, c_y, box_size)]

    for [w1, w2, w3, w4] in wall_set:
        mat[w1[0]:w1[1], w1[2]] = 1.
        mat[w2[0]:w2[1], w2[2]] = 1.
        mat[w3[0], w3[1], w3[2]] = 1.
        mat[w4[0], w4[1], w4[2]] = 1.

    return mat


def hm_corridor_various_width(res):
    # Make even dimensions
    M = math.ceil(res * 10) * 2
    N = math.ceil(res * 100) * 2
    mat = np.zeros((M, N), dtype=np.float32)

    N_segments = 5
    min_width, max_width, min_length, max_length = (40, 80, 40, 80)
    wall_set = []

    c_x, c_y = (10, M / 2)

    def getbox(x, y, width, length):
        halfwidth = int(width / 2)
        halflength = int(length / 2)
        return (x - halflength, x + halflength, y - halfwidth), \
               (x - halflength, x + halflength, y + halfwidth), \
               (y - halfwidth, y + halfwidth, x - halflength), \
               (y - halfwidth, y + halfwidth, x + halflength)

    # Add first box
    [wall_set.add(w) for w in getbox(c_x, c_y, max_width, max_length)]

    for i in range(N_segments):
        width = np.random.randint(min_width, max_width)
        length = np.random.randint(min_width, max_width)

        # Add to wall_list while removing overlapping walls
        [wall_set.add(w) for w in getbox(c_x, c_y, width, length)]

    for [w1, w2, w3, w4] in wall_set:
        mat[w1[0]:w1[1], w1[2]] = 1.
        mat[w2[0]:w2[1], w2[2]] = 1.
        mat[w3[0], w3[1], w3[2]] = 1.
        mat[w4[0], w4[1], w4[2]] = 1.

    return mat


def hm_stairs(res):
    # Make even dimensions
    M = math.ceil(res * 10) * 2
    N = math.ceil(res * 100) * 2
    mat = np.zeros((M, N), dtype=np.float32)

    lim_asc = 1, 8
    lim_desc = 1, 6
    lim_flat = 1, 5

    N_steps = 20

    # Generate step sequence
    steps = []
    curr_height = 0
    for i in range(N_steps):
        c = np.random.choice(("u", "d", "f"))
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
    step_len, step_height = 15, 10

    # Buffer steps sequence initially with zeros
    steps.reverse()
    steps.extend([0] * 5)
    steps.reverse()

    # Fill in height map
    for i, s in enumerate(steps):
        mat[i * step_len: (i + 1) * step_len , :M] = s * step_height

    # Add walls
    max_height = max(steps) * step_height
    mat[:N_steps * step_len, 0] = max_height + 10
    mat[:N_steps * step_len, M] = max_height + 10

    return mat


def hm_pillars(res):
    # Make even dimensions
    tiling_res = 6
    M = math.ceil(res * 8) * tiling_res
    N = math.ceil(res * 30) * tiling_res
    mat = np.zeros((M, N), dtype=np.float32)

    pillar_size = 10
    padding = 5
    block_size = int(M / tiling_res)

    M_tiles = int(M / tiling_res)
    N_tiles = int(N / tiling_res)

    # Add blocks
    for i in range(M_tiles):
        for j in range(N_tiles):
            if np.random.randn() < 0.5: continue
            rnd_x = np.random.randint(j * block_size + padding, (j + 1) * block_size - pillar_size - padding)
            rnd_y = np.random.randint(i * block_size + padding, (i + 1) * block_size - pillar_size - padding)
            mat[i * block_size + rnd_x: i * block_size + rnd_x + pillar_size,
            j * block_size + rnd_y: j * block_size + rnd_y + pillar_size] = 1

    # Add walls
    mat[0, 0:] = 1.
    mat[:, 0:] = 1.
    mat[0:, 0] = 1.
    mat[0:, :] = 1.

    return mat


def hm_pipe(res, diameter):
    # Make even dimensions
    M = math.ceil(res * 10) * 2
    N = math.ceil(res * 100) * 2

    mat = 1 - np.abs(np.linspace(diameter * -np.pi/2, diameter *  np.pi/2, M).repeat(N).reshape(M, N))

    # Add walls
    mat[0, 0:] = 1.
    mat[:, 0:] = 1.
    mat[0:, 0] = 1.
    mat[0:, :] = 1.

    return mat


def hm_tunnel(res, diameter):

    # Make even dimensions
    M = math.ceil(res * 10) * 2
    N = math.ceil(res * 100) * 2

    # Base
    mat = np.ones((M, N), dtype=np.float32)

    # Generate trajectory points
    N_pts = 30
    step = np.floor(N / N_pts)
    curr_y, curr_z = M / 2, 0.3
    for i in range(1, N_pts):
        mat[curr_y - diameter * 10:curr_y + diameter * 10,step * i:step * i + step] = 1 - curr_z - np.abs(np.linspace(diameter * -np.pi/2, diameter *  np.pi/2, M).repeat(N).reshape(M, N))
        curr_y += np.random.rand() * 0.02 - 0.01
        curr_z += np.random.rand() * 0.02 - 0.01

    return mat


def img_generation():
    N = 150
    M = 30

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "assets/stairs.png")

    # Generate stairs
    mat = np.zeros((M, N))

    stair_height = 20
    stair_width = 3
    current_height = 0

    for i in range(6):
        mat[:, 10 + i * stair_width : 10 + i * stair_width + stair_width] = current_height
        current_height += stair_height

    for i in range(3):
        mat[:, 28 + i * stair_width :  28 + i * stair_width + stair_width] = current_height

    for i in range(4):
        mat[:, 37 + i * stair_width : 37 + i * stair_width + stair_width] = current_height
        current_height -= stair_height

    for i in range(2):
        mat[:, 49 + i * stair_width :  49 + i * stair_width + stair_width] = current_height

    for i in range(3):
        mat[:, 55 + i * stair_width: 55 + i * stair_width + stair_width] = current_height
        current_height -= stair_height

    #---
    for i in range(12):
        mat[:, 55 + 10 + i * stair_width : 55 + 10 + i * stair_width + stair_width] = current_height
        current_height += stair_height

    for i in range(15):
        mat[:, 70 + 28 + i * stair_width : 70 +  28 + i * stair_width + stair_width] = current_height


    mat[0, :] = 255
    mat[:, 0] = 255
    mat[-1, :] = 255
    mat[:, -1] = 255
    cv2.imwrite(filename, mat)


if __name__ == "__main__":
    pass