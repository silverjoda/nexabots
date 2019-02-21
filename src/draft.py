import cma
import numpy as np
import time

import cv2

mat = np.zeros((30,120), dtype=np.uint8)
mat[:,:2] = 255
mat[:,-2:] = 255
mat[:2,:] = 255
mat[-2:,:] = 255

cv2.imwrite("hf_gen.png", mat)