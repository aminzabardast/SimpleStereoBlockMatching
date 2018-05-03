import numpy as np


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
        self._disparity_map = np.zeros(shape=self._left_image.shape)

    def _block_matching_for_single_chanel_image(self):
        """Algorithm for single channel images"""

        # Handling inputs and their defaults:
        # Defining the minimum and the maximum radius for kernel
        min_radius = self._kwargs['min_block_size']//2 if 'min_block_size' in self._kwargs.keys() else 2
        max_radius = self._kwargs['max_block_size']//2 if 'max_block_size' in self._kwargs.keys() else 10
        # Threshold for variance
        var_threshold = self._kwargs['var_threshold'] if 'var_threshold' in self._kwargs.keys() else 2
        # Disparity Range
        disparity_range = self._kwargs['disparity_range'] if 'disparity_range' in self._kwargs.keys() else 16

        # Left Image Dimensions
        rows, columns = self._left_image.shape
        # Iterating rows
        for i in range(0, rows):
            # Iterating Columns
            for j in range(0, columns):
                # Defining the minimum radius for kernel
                local_min_radius = int(min_radius)
                # Extracting left block
                # If the variance of pixels is lower than a threshold, then the radius of the kernel will be
                # increased so more details get in the window.
                # This is effective against homogeneous regions
                left_block = self._extract_block(self._left_image, i, j, local_min_radius)
                while left_block.var() < var_threshold and local_min_radius < max_radius:
                    local_min_radius += 1
                    left_block = self._extract_block(self._left_image, i, j, local_min_radius)
                # Error Array
                errors = np.ndarray(shape=(0,))
                # Searching for match in the right image
                # The match will be in the left side of the current poisson
                for k in range(j, j-disparity_range, -1):
                    # Extracting right block
                    right_block = self._extract_block(self._right_image, i, k, local_min_radius)
                    # If the window sizes does not match, this means that we reached the end of the row
                    if left_block.shape != right_block.shape:
                        break
                    # Calculating Error
                    errors = np.append(errors, self._mse(left_block, right_block))
                # Minimum Index
                min_idx = int(np.where(errors == errors.min())[0][0])
                # Interpolating the result with a parabola to gain sub-pixel accuracy.
                if min_idx == 0 or min_idx == len(errors)-1:
                    self._disparity_map[i, j] = min_idx
                else:
                    self._disparity_map[i, j] = min_idx + 0.5 * (errors[min_idx-1]-errors[min_idx+1]) / \
                        (errors[min_idx-1]-2*errors[min_idx]+errors[min_idx+1])
        return True

    @staticmethod
    def _extract_block(img, i, j, radius):
        """Extracting a window from a image"""
        x0 = 0 if i - radius < 0 else i - radius
        x1 = img.shape[0] if i + radius >= img.shape[0] else i + radius + 1
        y0 = 0 if j - radius < 0 else j - radius
        y1 = img.shape[1] if j + radius >= img.shape[1] else j + radius + 1
        return img[x0:x1, y0:y1]

    @staticmethod
    def _mse(left_block, right_block):
        """Calculating 'Mean Squared Error"""
        return np.sum(np.power(left_block - right_block, 2))

    @staticmethod
    def _mae(left_block, right_block):
        """Calculating 'Mean Absolute Error'"""
        return np.sum(np.abs(left_block - right_block)) / left_block.shape[0]*left_block.shape[1]

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
        self._occlusion_penalty = self._kwargs['occlusion_penalty'] if 'occlusion_penalty' in self._kwargs.keys() else 10
        self.show_occlusions = self._kwargs['show_occlusions'] if 'show_occlusions' in self._kwargs.keys() else False

        # Sizing the images
        rows, columns = self._left_image.shape

        # Optimizing each row using Dynamic Programming
        for i in range(0, rows-2):
            # Creating optimization graph
            self._create_matching_grid(columns)
            # Optimizing
            self._dp_cost(i, 2, 2)
            self._disparity_map[i, :] = self._return_dp_shortest_path()
        return True

    def _create_matching_grid(self, length):
        """Creating a numpy array as optimization graph"""
        self._matching_grid = np.full(shape=(length, length), fill_value=np.inf)
        return True

    def _dp_cost(self, row, a, b):
        """Calculating Stereo Matching cost graph"""

        # Memoization for faster respond
        if self._matching_grid[a, b] < np.inf:
            return self._matching_grid[a, b]

        # Base cases
        # If last node have a direct connection
        if a == self._right_image.shape[1]-3 and b == self._left_image.shape[1]-3:
            self._matching_grid[a, b] = self._mse(self._right_image[row-2:row+2, a-2:a+2],
                                                  self._left_image[row-2:row+2, b-2:b+2])
        # If last node have a right connection
        elif a == self._right_image.shape[1]-3:
            self._matching_grid[a, b] = self._dp_cost(row, a, b+1) + self._occlusion_penalty + \
                                        self._mse(self._right_image[row-2:row+2, a-2:a+2],
                                                  self._left_image[row-2:row+2, b-2:b+2])
        # If last node have a left connection
        elif b == self._left_image.shape[1]-3:
            self._matching_grid[a, b] = self._dp_cost(row, a+1, b) + self._occlusion_penalty + \
                                        self._mse(self._right_image[row-2:row+2, a-2:a+2],
                                                  self._left_image[row-2:row+2, b-2:b+2])
        # Recursion
        else:
            self._matching_grid[a, b] = np.min([
                self._dp_cost(row, a + 1, b + 1),
                self._dp_cost(row, a, b + 1) + self._occlusion_penalty,
                self._dp_cost(row, a + 1, b) + self._occlusion_penalty,
            ]) + self._mse(self._right_image[row-2:row+2, a-2:a+2], self._left_image[row-2:row+2, b-2:b+2])
        return self._matching_grid[a, b]

    def _return_dp_shortest_path(self):
        """Returning the shortest path in graph"""

        # Starting from first pixel
        i, j = 0, 0
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
            if self.show_occlusions and temp == i:
                continue
            else:
                disparity[j] = np.abs(i-j)
                temp = i
        return disparity
