import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# Load holes
holes = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"policies/RNN_['holes']_K77.npy"))
holes_av = np.mean(holes, 0)
holes_av = savgol_filter(holes_av, 501, 2)

# Load pipe
pipe = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "policies/RNN_['pipe']_R2D.npy"))
pipe_av = np.mean(pipe, 0)
pipe_av = savgol_filter(pipe_av, 501, 2)

# Load holes-pipe
holes_pipe = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "policies/RNN_['holes', 'pipe']_W9D.npy"))
holes_pipe_av = np.mean(holes_pipe, 0) / 1.
holes_pipe_av = savgol_filter(holes_pipe_av, 501, 2)

# Holes-pipe expert average
holes_pipe_expert_av = (holes_av + pipe_av) / 2

#fig = plt.figure()

t = np.arange(0, 7500)

# red dashes, blue squares and green triangles
plt.plot(t, holes_av, 'r-', label='holes')
plt.plot(t, pipe_av, 'b-', label='pipe')
plt.plot(t, holes_pipe_expert_av, 'g-', label='holes_pipe_average_experts')
plt.plot(t, holes_pipe_av, 'k-', label='holes_pipe_average')
plt.xlabel('Iters')
plt.ylabel('Score')
plt.legend()
plt.show()