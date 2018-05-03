import matplotlib.pyplot as plt
import numpy as np
from SimpleSM.matching import StereoMatcher
from time import time

# Importing Images and converting to gray scale
imgL = plt.imread('images/left.ppm')
imgL = np.dot(imgL[..., :], [.333, .333, .334])
imgR = plt.imread('images/right.ppm')
imgR = np.dot(imgR[..., :], [.333, .333, .334])

# Running the algorithm
matcher = StereoMatcher(imgL, imgR)

# Computing main method + run time
start_time = time()
matcher.compute()
end_time = time()
print('Runtime is: {} s'.format(end_time-start_time))

# Saving the image
disparity_map = matcher.get_disparity_image()
plt.imsave('result.png', disparity_map, cmap=plt.get_cmap('jet'))
