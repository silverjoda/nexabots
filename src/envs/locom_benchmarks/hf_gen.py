import numpy as np
import cv2
import os
import math
from math import exp
import noise
from scipy.misc import toimage

def hm_flat(res):
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.zeros((M, N), dtype=np.float32)

    # Walls
    mat[0, :] = 1.
    mat[-1, :] = 1.
    mat[:, 0] = 1.
    mat[:, -1] = 1.

    return mat


def hm_corridor(res, cw=8):
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.zeros((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    # Walls
    mat[M_2 - cw : M_2 + cw, 0] = 1.
    mat[M_2 - cw : M_2 + cw, -1] = 1.
    mat[M_2 - cw, :] = 1.
    mat[M_2 + cw, :] = 1.
    return mat


def hm_corridor_holes(res, cw=8):
    # Make even dimensions
    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.ones((M, N), dtype=np.float32)
    M_2 = math.ceil(M / 2)

    # Amount of 'tiles'
    Mt = 2
    Nt = 25

    # Makes tiles array
    p_kill = 0.5
    tiles_array = np.ones((2, Nt)) * 0.4
    for i in range(Nt):
        if np.random.rand() < p_kill:
            tiles_array[np.random.randint(0,2), i] = 0

    # Translate tiles array to real heightmap
    for i in range(Nt):
        mat[M_2 - cw: M_2, i * cw: i * cw + cw] = tiles_array[0, i]
        mat[M_2:M_2 + cw:, i * cw: i * cw + cw] = tiles_array[1, i]

    # Walls
    mat[:, 0] = 1.
    mat[:, -1] = 1.

    # Flat starting pos
    mat[M_2 - cw: M_2 + cw, : 4 * cw] = 0.4

    # Multiply to full image resolution
    mat *= 255

    return mat


def hm_corridor_turns(res):
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


    # Add initial wall
    # mat[M_2 - cw: M_2 + cw, 0] = 1.

    return mat


def hm_corridor_various_width(res):
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


    return mat


def hm_pillars_random(res):
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

    return mat


def hm_pillars_pseudorandom(res):
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

    return mat


def hm_tiles(res):
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

    return mat


def hm_triangles(res):
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
    mat[:, 0] = 1.
    mat[:, -1] = 1.

    # Multiply to full image resolution
    mat *= 255

    return mat


def hm_domes(res):
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

    return mat


def hm_stairs(res):
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

    return mat


def hm_pipe(res, radius=8):
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

    return mat


def hm_pipe_variable_rad(res, min_rad=6, max_rad=12):
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


def hm_verts(res):
    # Make even dimensions
    M = math.ceil(res * 10) * 2
    N = math.ceil(res * 100) * 2

    wdiv = 4
    ldiv = 14
    mat = np.random.rand((M // wdiv, N // ldiv)).repeat(wdiv, axis=0).repeat(ldiv, axis=1)
    mat[:, :50] = 0
    mat[mat < 0.5] = 0
    mat = 1 - mat

    return mat


def hm_perlin(res, scale_x, scale_y, base, octaves=3, persistence=0.5, lacunarity=2.0):
    shape = (res, res)
    scale_x = scale_x
    scale_y = scale_y

    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i / scale_x,
                                        j / scale_y,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=256,
                                        repeaty=256,
                                        base=base)




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
    from math import exp
    import numpy as np
    import matplotlib.pyplot as plt


    def rbf_kernel(x1, x2, variance=1):
        return exp(-1 * ((x1 - x2) ** 2) / (2 * variance))


    def gram_matrix(xs):
        return [[rbf_kernel(x1, x2) for x2 in xs] for x1 in xs]


    xs = np.arange(0, 20, 0.1)
    mean = [0 for x in xs]
    gram = gram_matrix(xs)

    plt_vals = []
    for i in range(0, 5):
        ys = np.random.multivariate_normal(mean, gram)
        plt_vals.extend([xs, ys, "k"])
    plt.plot(*plt_vals)
    plt.show()