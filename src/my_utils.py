import torch as T
import numpy as np
import math

def to_tensor(x, add_batchdim=False):
    x = T.FloatTensor(x.astype(np.float32))
    if add_batchdim:
        x = x.unsqueeze(0)
    return x

def quat_to_rpy(quat):
    qw, qx, qy, qz = quat

    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if (np.abs(sinp) >= 1):
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

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