import numpy as np
from skimage.feature import match_template


class StereoMatcher:
    def __init__(self, left_image, right_image, method='dynamic_programming', **kwargs):
        if left_image.shape != right_image.shape:
            raise IndexError('Left and Right images do not have the same dimensions.')

        # Saving method related arguments
        self._kwargs = kwargs

        # Method
        self._method = method

        # Saving Variables into Attributes
        self._left_image = left_image
        self._right_image = right_image

        # Disparity
        self._disparity_map = np.full(shape=self._left_image.shape, fill_value=np.nan)

    def _block_matching_for_single_chanel_image(self):
        """Algorithm for single channel images"""

        # Handling inputs and their defaults:
        # Defining the minimum and the maximum radius for kernel
        kernel_radius = self._kwargs['kernel_size']//2 if 'kernel_size' in self._kwargs.keys() else 2  # 5
        # Disparity Range
        disparity_range = self._kwargs['disparity_range'] if 'disparity_range' in self._kwargs.keys() else 16

        # Left Image Dimensions
        rows, columns = self._left_image.shape
        # Iterating rows
        for i in range(kernel_radius, rows - kernel_radius):
            # Iterating Columns
            for j in range(disparity_range + kernel_radius, columns - kernel_radius):
                # Extracting left block
                # If the variance of pixels is lower than a threshold, then the radius of the kernel will be
                # increased so more details get in the window.
                # This is effective against homogeneous regions
                left_block = self._extract_block(self._left_image, i, j, kernel_radius)
                # Searching for match in the right image
                # The match will be in the left side of the current poisson
                search_space = self._extract_row(self._right_image, i, j, kernel_radius, disparity_range)
                # Calculating Max index in search space (Cross Correlation)
                cross_correlation = match_template(image=search_space, template=left_block)
                # Finding the max
                max_idx = np.argmax(cross_correlation, axis=1)
                cross_correlation = np.sort(cross_correlation, axis=1)[0]
                # Calculating disparity
                self._disparity_map[i, j] = disparity_range - max_idx - 1
        return True

    @staticmethod
    def _extract_block(img, i, j, radius):
        """Extracting a window from a image"""
        return img[i - radius:i + radius + 1, j - radius:j + radius + 1]

    @staticmethod
    def _extract_row(img, i, j, radius, disparity):
        """Extracting a window from a image"""
        return img[i - radius:i + radius + 1, j - disparity - radius + 1:j + radius + 1]

    @staticmethod
    def _ssd(left_block, right_block):
        """Calculating 'Mean Squared Error"""
        return np.sum(np.power(left_block - right_block, 2))

    def compute(self):
        """Main Simple Block Matching Algorithm / Input images should be rectified"""
        # Gray Scale Images
        if len(self._left_image.shape) != 2:
            raise TypeError('The image is not Gray Scale.')

        # Simple block Matching
        if self._method == 'block_matching':
            return self._block_matching_for_single_chanel_image()
        elif self._method == 'dynamic_programming':
            return self._dynamic_programming_for_single_chanel_image()

        # Incorrect method
        raise ValueError('Calculation method is not correct')

    def get_disparity(self):
        """Returning Disparity matrix"""
        return self._disparity_map if self._disparity_map.any() else False

    def get_disparity_image(self):
        """Return Disparity suited for plotting"""
        if not self._disparity_map.any():
            return False
        min_value = self._disparity_map[np.isnan(self._disparity_map) == False].min()-1
        self._disparity_map[np.isnan(self._disparity_map) == True] = min_value
        return self._disparity_map

    def _dynamic_programming_for_single_chanel_image(self):

        # Handling inputs and their defaults:
        self._occlusion_penalty = self._kwargs['occlusion_penalty'] if 'occlusion_penalty' in self._kwargs.keys() else 0
        self._show_occlusions = self._kwargs['show_occlusions'] if 'show_occlusions' in self._kwargs.keys() else False
        disparity_range = self._kwargs['disparity_range'] if 'disparity_range' in self._kwargs.keys() else 16

        # Sizing the images
        rows, columns = self._left_image.shape

        # Optimizing each row using Dynamic Programming
        for i in range(0, rows):
            # Creating optimization graph
            self._create_matching_grid(columns)
            # Optimizing (Using recursion)
            self._dp_cost(i, 0, 0, disparity_range)
            # Calculating disparity information
            self._disparity_map[i, :] = self._return_dp_shortest_path(0, 0)
        return True

    def _create_matching_grid(self, length):
        """Creating a numpy array as optimization graph"""
        self._matching_grid = np.full(shape=(length, length), fill_value=np.inf)
        return True

    def _dp_cost(self, row, a, b, max_disparity):
        """Calculating Stereo Matching cost graph"""

        # Not checking pixels further than disparity range
        # This reduces complexity of the model
        if np.abs(a-b) > max_disparity:
            return np.inf

        # Memoization for faster respond
        if self._matching_grid[a, b] < np.inf:
            return self._matching_grid[a, b]

        # Base cases
        # If last node have a direct connection
        if a == self._right_image.shape[1]-1 and b == self._left_image.shape[1]-1:
            self._matching_grid[a, b] = self._ssd(self._right_image[row, a],
                                                  self._left_image[row, b])
        # If last node have a right connection
        elif a == self._right_image.shape[1]-1:
            self._matching_grid[a, b] = self._dp_cost(row, a, b+1, max_disparity) + self._occlusion_penalty + \
                                        self._ssd(self._right_image[row, a],
                                                  self._left_image[row, b])
        # If last node have a left connection
        elif b == self._left_image.shape[1]-1:
            self._matching_grid[a, b] = self._dp_cost(row, a+1, b, max_disparity) + self._occlusion_penalty + \
                                        self._ssd(self._right_image[row, a],
                                                  self._left_image[row, b])
        # Recursion
        else:
            self._matching_grid[a, b] = np.min([
                self._dp_cost(row, a + 1, b + 1, max_disparity),
                self._dp_cost(row, a, b + 1, max_disparity) + self._occlusion_penalty,
                self._dp_cost(row, a + 1, b, max_disparity) + self._occlusion_penalty,
            ]) + self._ssd(self._right_image[row, a], self._left_image[row, b])
        return self._matching_grid[a, b]

    def _return_dp_shortest_path(self, i, j):
        """Returning the shortest path in graph"""

        shortest_path = []
        while i < self._matching_grid.shape[0]-1 and j < self._matching_grid.shape[1]-1:
            # Adding Node to shortest path list
            shortest_path.append([i, j])
            # Calculating next node
            min_cost = np.min([
                self._matching_grid[i+1, j+1],
                self._matching_grid[i, j+1],
                self._matching_grid[i+1, j]
            ])
            if min_cost == self._matching_grid[i+1, j+1]:
                i += 1
                j += 1
            elif min_cost == self._matching_grid[i, j+1]:
                j += 1
            elif min_cost == self._matching_grid[i+1, j]:
                i += 1
        # Adding last node into path manually
        shortest_path.append([self._matching_grid.shape[0]-1, self._matching_grid.shape[0]-1])

        # Calculating disparity from shortest path
        disparity = np.full(shape=(self._matching_grid.shape[0],), fill_value=np.nan)
        temp = np.nan
        for i, j in shortest_path:
            if self._show_occlusions and temp == i:
                continue
            else:
                disparity[j] = np.abs(i-j)
                temp = i
        return disparity
