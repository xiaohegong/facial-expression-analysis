import cv2 as cv
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
from model.vgg16 import VGG16
import torch as T

def upsampling(image, target_width, target_height):
    # Require grey scale image
    dim = (target_width, target_height)
    resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    return resized

def load_data(images, label):
    n = images.shape[0]
    result = []
    for i in range(n):
        upsampled = upsampling(images[i], 224, 224)
        result.append([upsampled, label[i]])
    return np.array(result)

if __name__ == '__main__':
    datapath = os.path.join("data", "CK_data.h5")
    f = h5py.File(datapath, "r")
    images = np.array(f['data_samples'])
    labels = np.array(f['data_labels'])

    dataset = load_data(images, labels)
    np.random.shuffle(dataset)
    len = dataset.shape[0]
    spliter = np.split(dataset, [int(np.floor(len * 0.7)), int(len)])

    model = VGG16(alpha=0.01, epochs=25, batch_size=1, dataset=spliter, num_classes=7)
    model._train()

    dst_path = "model_data/vgg16_ck.pt"
    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))

    T.save(model, dst_path)
