import numpy as np


class BlockMatcher:
    def __init__(self, left_image, right_image, block_size=5, disparity_range=16):
        if left_image.shape != right_image.shape:
            raise IndexError('Left and Right images do not have the same dimensions.')

        if block_size < 1 or block_size % 2 == 0:
            raise IndexError('Kernel size is not a positive odd number.')

        self._leftImage = left_image
        self._rightImage = right_image
        self._blockSize = block_size
        self._disparityRange = disparity_range

        # Initial disparity
        self._disp = np.zeros(shape=self._leftImage.shape)

    def _compute_for_single_channel_image(self):
        """Algorithm for single channel images"""
        # Left Image Dimensions
        rows, columns = self._leftImage.shape

        # Avoiding Index Errors
        margin = self._blockSize // 2
        # Iterating rows
        for i in range(margin, rows - margin):
            # Iterating Columns
            for j in range(margin, columns - margin):
                # Extracting left block
                left_block = self._leftImage[i - margin:i + margin + 1, j - margin:j + margin + 1]
                # Error Array
                errors = np.ndarray(shape=(0,))
                # End of search in the row, beginning from j
                search_end = margin - 1 if j - self._disparityRange < margin - 1 else j - self._disparityRange
                # Searching for match in the right image
                # The match will be in the left side of the current poisson
                for k in range(j, search_end, -1):
                    # Extracting right block
                    right_block = self._rightImage[i - margin:i + margin + 1, k - margin:k + margin + 1]
                    # Calculating Error
                    errors = np.append(errors, self._mse(left_block, right_block))
                # Minimum Index
                min_idx = int(np.where(errors == errors.min())[0][0])
                # Interpolating the result with a parabola to gain sub-pixel accuracy.
                if min_idx == 0 or min_idx == len(errors)-1:
                    self._disp[i, j] = min_idx
                else:
                    self._disp[i, j] = min_idx + 0.5 * (errors[min_idx-1]-errors[min_idx+1]) / \
                        (errors[min_idx-1]-2*errors[min_idx]+errors[min_idx+1])
        return True

    def _mse(self, left_block, right_block):
        """Calculating 'Mean Squared Error"""
        return np.sum(np.power(left_block - right_block, 2)) / self._blockSize ** 2

    def _mae(self, left_block, right_block):
        """Calculating 'Mean Absolute Error'"""
        return np.sum(np.abs(left_block - right_block)) / self._blockSize ** 2

    def compute(self):
        """Main Simple Block Matching Algorithm / Input images should be rectified"""
        # Gray Scale Images
        if len(self._leftImage.shape) == 2:
            self._compute_for_single_channel_image()
        # Not Gray Scale
        else:
            raise TypeError('The image is not Gray Scale.')

    def disparity(self):
        """Returning Disparity"""
        return self._disp
