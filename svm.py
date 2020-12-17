import h5py
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

def load_data(images, label):
    n = images.shape[0]
    ims = []
    labels = []
    for i in range(n):
        ims.append(images[i].flatten())
        labels.append((label[i]))
    return ims, labels

if __name__ == "__main__":
    datapath = os.path.join("data", "fer2013_data.h5")
    f = h5py.File(datapath, "r")
    images = np.array(f['data_samples'])
    labels = np.array(f['data_labels'])

    # extract data into correct format: list of [input, label]
    images, labels = load_data(images, labels)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3)

    # np.random.shuffle(dataset)
    # len = dataset.shape[0]
    # spliter = np.split(dataset, [int(np.floor(len * 0.7)), int(len)])

    model = svm.SVC(C=1.0, kernel='rbf', degree=3)
    model.fit(x_train, y_train)
    print(model.score(x_train, y_train))
    predictions = model.predict(x_test)
    print(accuracy_score(predictions, y_test))