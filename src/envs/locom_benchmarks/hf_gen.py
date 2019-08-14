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
    pass


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