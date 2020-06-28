import numpy as np
import matplotlib.pyplot as plt

chess = np.zeros((10,5))
chess[1::2, 0::2] = 1
chess[0::2, 1::2] = 1

plt.imshow(chess, cmap = 'binary')
plt.show()
