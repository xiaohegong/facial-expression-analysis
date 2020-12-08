import h5py
import numpy as np
import os

import matplotlib.pyplot as plt

import torch as T
from model.cnn import CNN


if __name__ == "__main__":
    datapath = os.path.join("data", "CK_bipart_data.h5")
    f = h5py.File(datapath, "r")
    eyes = np.array(f['mouth_samples'])
    mouths = np.array(f['mouth_samples'])
    labels = np.array(f['data_labels'])

    # extract data into correct format: list of [input, label]
    #dataset = load_data(images, labels)
    # np.random.shuffle(dataset)
    # len = dataset.shape[0]
    # spliter = np.split(dataset, [int(np.floor(len * 0.7)), int(len)])
    #
    # model = CNN(alpha=0.001, epochs=10, batch_size=128, dataset=spliter, num_classes=7)
    # model._train()
    #
    # dst_path = "model_data/cnn.pt"
    # if not os.path.exists(os.path.dirname(dst_path)):
    #     os.makedirs(os.path.dirname(dst_path))
    #
    # T.save(model, dst_path)