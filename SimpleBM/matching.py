import numpy as np


class BlockMatcher:
    def __init__(self, left_image, right_image, method='block_matching', **kwargs):
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

    def _compute_for_single_channel_image(self):
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
        return np.sum(np.power(left_block - right_block, 2)) / left_block.shape[0]*left_block.shape[1]

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
            return self._compute_for_single_channel_image()

        # Incorrect method
        raise ValueError('Calculation method is not correct')

    def get_disparity(self):
        return self._disparity_map if self._disparity_map.any() else False
