import h5py
import numpy as np
import os

import matplotlib.pyplot as plt

from model.cnn import CNN


def load_data(images, label):
    n = images.shape[0]
    result = []
    for i in range(n):
        result.append([images[i], label[i]])
    return np.array(result)

if __name__ == "__main__":
    datapath = os.path.join("data", "CK_data.h5")
    f = h5py.File(datapath, "r")
    images = np.array(f['data_samples'])
    labels = np.array(f['data_labels'])

    # extract data into correct format: list of [input, label]
    dataset = load_data(images, labels)
    model = CNN(alpha=0.001, epochs=10, batch_size=128, dataset=dataset, num_classes=7)
    model._train()
    model.save(model, "model_data/cnn.pt")