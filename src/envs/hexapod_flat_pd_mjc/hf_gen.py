import numpy as np
import cv2
import os

filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "assets/hf2.png")

N = 120
M = 30
div = 5
mat = np.random.randint(0, 70, size=(M//div,N//div), dtype=np.uint8).repeat(div, axis=0).repeat(div, axis=1)
mat[0,:] = 255
mat[:,0] = 255
mat[-1,:] = 255
mat[:,-1] = 255
#mat[N//2 - 4: N//2 + 4, N//2 - 4: N//2 + 4] = 255
cv2.imwrite(filename, mat)


