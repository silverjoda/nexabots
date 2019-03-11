import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
# bumps = np.load(filepath+ "bumps_states.npy")
# flat = np.load(filepath+ "flat_states.npy")
# holes = np.load(filepath+ "holes_states.npy")
rails = np.load(filepath+ "rails_states.npy")
# stairs = np.load(filepath+ "stairs_states.npy")
# tiles = np.load(filepath+ "tiles_states.npy")

s_dim = rails.shape[-1]

sets = (rails)
#X = np.concatenate(sets)
X = rails[:,:,-1,:]
plt.imshow(X[0].T)
plt.show()
exit()

X_flat = X.reshape((-1, s_dim))
X_embedded = TSNE(n_components=2).fit_transform(X_flat)

#c = np.concatenate([[i] * 200 for i in range(6)])
c = np.concatenate([np.arange(200) for i in range(1)])

pts_X = X_embedded[0:200, 0]
pts_Y = X_embedded[0:200, 1]
plt.scatter(pts_X, pts_Y, c=c)
plt.show()
