import torch as T
import numpy as np
import math

def to_tensor(x, add_batchdim=False):
    x = T.FloatTensor(x.astype(np.float32))
    if add_batchdim:
        x = x.unsqueeze(0)
    return x

def quat_to_rpy(quat):
    q0, q1, q2, q3 = quat
    roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 - q2 ** 2))
    pitch = math.asin(2 * (q0 * q2 - q3 * q1))
    yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 - q3 ** 2))

    return roll, pitch, yaw

def rpy_to_quat(roll, pitch, yaw ):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q0 = cy * cp * cr + sy * sp * sr
    q1 = cy * cp * sr - sy * sp * cr
    q2 = sy * cp * sr + cy * sp * cr
    q3 = sy * cp * cr - cy * sp * sr

    return (q0, q1, q2, q3)