import h5py
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

import torch as T
from model.cnn_fer import CNNByParts

CK_DATA = "CK_bipart_data.h5"
CK_DATA_HOG = "CK_bipart_hog.h5"
FER13_DATA = "fer13_bipart_data.h5"

def load_data(images, label):

    n = images.shape[0]
    result = []
    for i in range(n):
        result.append([images[i], label[i]])
    return np.array(result, dtype=object)

def shuffle_data(dataset):
    np.random.shuffle(dataset)
    N = dataset.shape[0]
    return np.split(dataset, [int(np.floor(N * 0.7)), int(N)])


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

if __name__ == "__main__":
    #datapath = os.path.join("data", CK_DATA_HOG)
    datapath = os.path.join("data", FER13_DATA)

    f = h5py.File(datapath, "r")
    mouths = np.array(f['mouth_samples'])
    eyes = np.array(f['eyes_samples'])
    # -------- For CK+ ---------
    # hog_eyes = np.array(f['hog_eyes'])
    # hog_mouth = np.array(f['hog_mouth'])
    # -----------------------------
    labels = np.array(f['data_labels'])

    # extract data into correct format: list of [input, label]
    mouth_dataset = load_data(mouths, labels)
    eyes_dataset = load_data(eyes, labels)
    # -------- For CK+ ---------
    # hog_eyes_dataset = load_data(hog_eyes, labels)
    # hog_mouth_dataset = load_data(hog_mouth, labels)
    # ---------------------------

    result = []
    for i in range(len(mouth_dataset)):
        sample = []
        sample.append(mouth_dataset[i])
        sample.append(eyes_dataset[i])

        # -----  For Fer2013 ----
        sample.append(np.array([get_hog(eyes_dataset[i][0]), labels[i]], dtype=object))
        sample.append(np.array([get_hog(mouth_dataset[i][0]), labels[i]], dtype=object))

        # ----- For CK+ -----
        # sample.append(hog_eyes_dataset[i])
        # sample.append(hog_mouth_dataset[i])
        result.append(np.array(sample))

    result = np.array(result)
    print(result.shape)

    model = CNNByParts(alpha=0.001, epochs=20, batch_size=128,
                       dataset=shuffle_data(result), num_classes=7)
    model._train()

    dst_path = "model_data/cnn_by_parts_fer13.pt"
    # dst_path = "model_data/cnn_by_parts_CK+.pt"
    # dst_path = "model_data/hog_by_parts_CK+.pt"
    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))

    T.save(model, dst_path)