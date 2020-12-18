import cv2 as cv
import numpy as np


def cell_gradient(cell_magnitude, cell_angle):
    bin_size = 6
    mag = [0] * bin_size
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_angle = cell_angle[k, l]
            if gradient_angle > 180:
                gradient_angle -= 180
            if gradient_angle >= 15 and gradient_angle < 45:
                mag[1] += cell_magnitude[k, l]
            elif gradient_angle >= 45 and gradient_angle < 75:
                mag[2] += cell_magnitude[k, l]
            elif gradient_angle >= 75 and gradient_angle < 105:
                mag[3] += cell_magnitude[k, l]
            elif gradient_angle >= 105 and gradient_angle < 135:
                mag[4] += cell_magnitude[k, l]
            elif gradient_angle >= 135 and gradient_angle < 165:
                mag[5] += cell_magnitude[k, l]
            else:
                mag[0] += cell_magnitude[k, l]
    return mag


def get_hog(img):
    n, m = img.shape

    Ix = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    Iy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    grad_magnitude = cv.addWeighted(Ix, 0.5, Iy, 0.5, 0)

    # Threshold
    for i in range(grad_magnitude.shape[0]):
        for j in range(grad_magnitude.shape[1]):
            if grad_magnitude[i, j] < 0:
                grad_magnitude[i, j] = 0
    gradient_angle = cv.phase(Ix, Iy, angleInDegrees=True)

    cell_size = 8
    bin_size = 6
    n = n // cell_size * cell_size
    m = m // cell_size * cell_size

    hog_mag = np.zeros((n // cell_size, m // cell_size, bin_size))

    # Get m x n x 6, 3D array store each
    for i in range(hog_mag.shape[0]):
        for j in range(hog_mag.shape[1]):
            cell_magnitude = grad_magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            hog_mag[i][j] = cell_gradient(cell_magnitude, cell_angle)
    return hog_mag
