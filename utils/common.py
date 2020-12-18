import numpy as np


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
