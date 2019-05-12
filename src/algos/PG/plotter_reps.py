import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load holes
holes = np.load("RNN_['holes']_D07.npy")
holes_av = np.mean(holes, 0)
holes_av = savgol_filter(holes_av, 501, 2)

# Load tiles
tiles = np.load("RNN_['tiles']_DK6.npy")
tiles_av = np.mean(tiles, 0)
tiles_av = savgol_filter(tiles_av, 501, 2)

# Load holes-tiles
holes_tiles = np.load("RNN_['tiles', 'holes']_B7B.npy")
holes_tiles_av = np.mean(holes_tiles, 0) / 1.
holes_tiles_av = savgol_filter(holes_tiles_av, 501, 2)

# Holes-tiles expert average
holes_tiles_expert_av = (holes_av + tiles_av) / 2

#fig = plt.figure()

t = np.arange(0, 7500)

# red dashes, blue squares and green triangles
plt.plot(t, holes_av, 'r-', label='holes')
plt.plot(t, tiles_av, 'b-', label='tiles')
plt.plot(t, holes_tiles_expert_av, 'g-', label='holes_tiles_average_experts')
plt.plot(t, holes_tiles_av, 'k-', label='holes_tiles_average')
plt.xlabel('Iters')
plt.ylabel('Score')
plt.legend()
plt.show()