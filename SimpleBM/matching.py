import numpy as np


class BlockMatcher:
    def __init__(self, left_image, right_image, disparity_range, var_threshold=2, max_block_size=21,
                 min_block_size=5):
        if left_image.shape != right_image.shape:
            raise IndexError('Left and Right images do not have the same dimensions.')

        # block sizes for dynamic window
        self._min_block_reduce = min_block_size//2
        self._max_block_reduce = max_block_size//2 if max_block_size else np.inf

        # Threshold for variance
        self._var_threshold = var_threshold

        # Saving Variables into Attributes
        self._left_image = left_image
        self._right_image = right_image
        self._disparity_range = disparity_range

    def _compute_for_single_channel_image(self):
        """Algorithm for single channel images"""
        # Left Image Dimensions
        rows, columns = self._left_image.shape

        # Initial disparity
        disparity_map = np.zeros(shape=self._left_image.shape)

        # Iterating rows
        for i in range(0, rows):
            # Iterating Columns
            for j in range(0, columns):
                # Defining the minimum radius for kernel
                min_radius = self._min_block_reduce
                # Extracting left block
                # If the variance of pixels is lower than a threshold, then the radius of the kernel will be
                # increased so more details get in the window.
                # This is effective against homogeneous regions
                left_block = self._extract_block(self._left_image, i, j, min_radius)
                while left_block.var() < self._var_threshold and min_radius < self._max_block_reduce:
                    min_radius += 1
                    left_block = self._extract_block(self._left_image, i, j, min_radius)
                # Error Array
                errors = np.ndarray(shape=(0,))
                # Searching for match in the right image
                # The match will be in the left side of the current poisson
                for k in range(j, j-self._disparity_range, -1):
                    # Extracting right block
                    right_block = self._extract_block(self._right_image, i, k, min_radius)
                    # If the window sizes does not match, this means that we reached the end of the row
                    if left_block.shape != right_block.shape:
                        break
                    # Calculating Error
                    errors = np.append(errors, self._mse(left_block, right_block))
                # Minimum Index
                min_idx = int(np.where(errors == errors.min())[0][0])
                # Interpolating the result with a parabola to gain sub-pixel accuracy.
                if min_idx == 0 or min_idx == len(errors)-1:
                    disparity_map[i, j] = min_idx
                else:
                    disparity_map[i, j] = min_idx + 0.5 * (errors[min_idx-1]-errors[min_idx+1]) / \
                        (errors[min_idx-1]-2*errors[min_idx]+errors[min_idx+1])
        return disparity_map

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
        return np.sum(np.abs(left_block - right_block))

    def compute(self):
        """Main Simple Block Matching Algorithm / Input images should be rectified"""
        # Gray Scale Images
        if len(self._left_image.shape) == 2:
            return self._compute_for_single_channel_image()
        # Not Gray Scale
        else:
            raise TypeError('The image is not Gray Scale.')
