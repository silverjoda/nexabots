import torch as T
import numpy as np

def to_tensor(x, add_batchdim=False):
    x = T.FloatTensor(x.astype(np.float32))
    if add_batchdim:
        x = x.unsqueeze(0)
    return x


def l_barron(x, a):
    return ((a - 2) / a) * (np.pow(np.square(x) / np.abs(a - 2), a/2) - 1)
