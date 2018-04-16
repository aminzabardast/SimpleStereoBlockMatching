import numpy as np

class BlockMatcher:
    def __init__(self, leftImage, rightImage, blockSize = 5, disparityRange = 16):
        if leftImage.shape != rightImage.shape:
            raise IndexError('Left and Right images do not have the same dimensions.')

        if blockSize < 1 or blockSize % 2 == 0:
            raise IndexError('Kernel size is not a positive odd number.')

        self._leftImage = leftImage
        self._rightImage = rightImage
        self._blockSize = blockSize
        self._disparityRange = disparityRange

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
                leftBlock = self._leftImage[i - margin:i + margin + 1, j - margin:j + margin + 1]
                # Initial Error
                minError = np.inf
                # End of search in the row, beginning from j
                asearchEnd = margin - 1 if j - self._disparityRange < margin - 1 else j - self._disparityRange
                # Searching for match in the right image
                # The match will be in the left side of the current poisson
                for k in range(j, asearchEnd, -1):
                    # Extracting right block
                    rightBlock = self._rightImage[i - margin:i + margin + 1, k - margin:k + margin + 1]
                    # Calculating Error
                    error = self._mse(leftBlock, rightBlock)
                    # Keeping the disparity with smallest error
                    if error < minError:
                        minError = error
                        self._disp[i, j] = j - k
        return True

    def _mse(self, leftBlock, rightBlock):
        """Calculating 'Mean Squared Error"""
        return np.sum(np.power(leftBlock - rightBlock, 2)) / self._blockSize ** 2

    def _mae(self, leftBlock, rightBlock):
        """Calculating 'Mean Absolute Error'"""
        return np.sum(np.abs(leftBlock - rightBlock)) / self._blockSize ** 2

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
