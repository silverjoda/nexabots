import numpy as np
import cv2
import os

N = 120
M = 30
div = 5

filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "assets/hf_rnd.png")

# Generate stairs
mat = np.random.randint(0, 70, size=(M // div, N // div), dtype=np.uint8).repeat(div, axis=0).repeat(div, axis=1)
# mat[:, 0:20] = 0
# mat[:, 40:60] = 0
# mat[:, 80:] = 0
#
# # FLAT !
# mat[:,:] = 0

mat[0, :] = 255
mat[:, 0] = 255
mat[-1, :] = 255
mat[:, -1] = 255
cv2.imwrite(filename, mat)

