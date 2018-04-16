import matplotlib.pyplot as plt
import numpy as np
from SimpleBM.matching import BlockMatcher

# Importing Images and converting to gray scale
imgL = plt.imread('images/left.ppm')
imgL = np.dot(imgL[..., :], [.333, .333, .334])
imgR = plt.imread('images/right.ppm')
imgR = np.dot(imgR[..., :], [.333, .333, .334])

# Disparity Template
disp = np.ndarray(shape=imgL.shape)

# Running the algorithm
matcher = BlockMatcher(left_image=imgL, right_image=imgR)
matcher.compute()

# Saving the image
disp[:, :] = matcher.disparity()/15
plt.imsave('result.png', disp)
