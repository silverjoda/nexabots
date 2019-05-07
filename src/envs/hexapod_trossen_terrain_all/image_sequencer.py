import numpy as np
import cv2
import os

imgs = []

for i in range(8):
    imgs.append(cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/s{}.png".format(i + 1))))

imgs = np.array(imgs)[[1,3,5,7]]
bg_mask = np.zeros_like(imgs[0], dtype=np.uint8)
master = np.zeros_like(imgs[0], dtype=np.uint8)

m,n,_ = bg_mask.shape
for i in range(m):
    for j in range(n):
        if len(np.unique(imgs[:, i, j, 0])) == 1 and len(np.unique(imgs[:, i, j, 1])) == 1 and len(np.unique(imgs[:, i, j, 2])) == 1:
            master[i, j, :] = imgs[0, i, j, :]
        else:
            uniques, counts = np.unique(imgs[:, i, j], return_counts=True, axis=0)
            fg = uniques[np.argmin(counts)]
            master[i, j, :] = fg

cv2.imshow('image',master)
cv2.waitKey(0)
cv2.destroyAllWindows()