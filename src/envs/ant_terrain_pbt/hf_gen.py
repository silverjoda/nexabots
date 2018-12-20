import numpy as np
import cv2

filename= "/home/silverjoda/SW/python-research/Hexaprom/src/envs/hf_l.png"

N = 640
M = 120
div = 20
mat = np.random.randint(0, 255, size=(M//div,N//div), dtype=np.uint8).repeat(div, axis=0).repeat(div, axis=1)
#mat[N//2 - 4: N//2 + 4, N//2 - 4: N//2 + 4] = 255
cv2.imwrite(filename, mat)


