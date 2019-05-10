import numpy as np
import matplotlib.pyplot as plt

# Load holes
holes = np.zeros((5,7500)) #np.load("fname")
holes_av = np.mean(holes, 0)

# Load pipe
pipe = np.zeros((5,7500)) #np.load("fname")
pipe_av = np.mean(pipe, 0)

# Load holes-pipe
holes_pipe = np.zeros((5,7500)) #np.load("fname")
holes_pipe_av = np.mean(holes_pipe, 0)

# Holes-pipe expert average
holes_pipe_expert_av = (holes_av + pipe_av) / 2

#fig = plt.figure()

t = np.arange(0, 7500)

# red dashes, blue squares and green triangles
plt.plot(t, holes_av, 'r-', label='holes')
plt.plot(t, pipe_av, 'b-', label='pipe')
plt.plot(t, holes_pipe_expert_av, 'k-', label='holes_pipe_average_experts')
plt.plot(t, holes_pipe_av, 'g-', label='holes_pipe_average')
plt.xlabel('Iters')
plt.ylabel('Score')
plt.legend()
plt.show()