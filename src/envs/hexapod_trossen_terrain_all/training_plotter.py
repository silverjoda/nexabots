import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter
import os

# Load holes
holes = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"policies_2/RNN_['holes']_V72.npy"))
holes_av = np.mean(holes, 0)
holes_av = savgol_filter(holes_av, 501, 2)

# Load pipe
pipe = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "policies_2/RNN_['pipe']_NRB.npy"))
pipe_av = np.mean(pipe, 0)
pipe_av = savgol_filter(pipe_av, 501, 2)

# Load holes-pipe
holes_pipe = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "policies_2/RNN_['holes', 'pipe']_7BO.npy"))
holes_pipe_av = np.mean(holes_pipe, 0) / 1.
holes_pipe_av = savgol_filter(holes_pipe_av, 501, 2)

# Holes-pipe expert average
holes_pipe_expert_av = (holes_av + pipe_av) / 2

#fig = plt.figure()

t = np.arange(0, 15000 * 400, 400)
t = np.linspace(0, 7, 15000)
matplotlib.axes.Axes.set_xscale
# red dashes, blue squares and green triangles
plt.plot(t, holes_av, 'r-', label='holes')
plt.plot(t, pipe_av, 'b-', label='pipe')
plt.plot(t, holes_pipe_expert_av, 'g-', label='holes + pipe experts average')
plt.plot(t, holes_pipe_av, 'k-', label='holes + pipe RNN average')
plt.xlabel(r'$\times 10^6$ steps')
plt.xlim([0,7])
plt.ylabel('Score')

plt.grid()
plt.legend()
plt.show()