"""
This file contains all training part of our program. It uses the models implemented in /models/ and save
the trained model to /model_data/ in .pt format.
"""

import matplotlib.pyplot as plt
import h5py
import os
from models.vgg16 import VGG16
from models.dcnn_model import CustomizedCNNModel
from models.CNN_by_parts import CNNByParts
from utils.hog_features import *
import torch as torch
from utils.common import load_data, shuffle_data

# Decide which dataset we gonna use
CK_DATA = "CK_bipart_data.h5"
CK_DATA_HOG = "CK_bipart_hog.h5"
FER13_DATA = "fer13_bipart_data.h5"


def train_bipart_cnn():
    """
    Train bipart model of CK+/Fer2013 and save them into .pt format.
    This is the CNN model A in our report
    :return: Right now it produces the bipart model for fer2013 dataset. If you want to train CK+,
    simply follow the comments below.
    """
    # datapath = os.path.join("data", CK_DATA_HOG)
    datapath = os.path.join("data", FER13_DATA)

    f = h5py.File(datapath, "r")
    mouths = np.array(f['mouth_samples'])
    eyes = np.array(f['eyes_samples'])
    # -------- For CK+ ----------
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
        # ------------------------

        # ----- For CK+ --------
        # sample.append(hog_eyes_dataset[i])
        # sample.append(hog_mouth_dataset[i])
        # --------------------
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

    torch.save(model, dst_path)


def train_dcnn_fer():
    """
    This our deep CNN model, which is the CNN model B in our report
    :return: A FER2013 model in .pt format
    """
    datapath = "data/fer2013_data.h5"

    f = h5py.File(datapath, "r")
    input = np.array(f['data_samples'])
    labels = np.array(f['data_labels'])
    dataset = load_data(input, labels)
    result = np.array(dataset)

    # Input parameters
    NUM_EPOCHS = 200
    model = CustomizedCNNModel(alpha=0.0001, epochs=NUM_EPOCHS, batch_size=128,
                               dataset=shuffle_data(result), num_classes=7)
    model._train()

    dst_path = "model_data/cnn_fer2013.pt"
    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))

    torch.save(model, dst_path)

    # Plot
    plt.plot(model.epochs_record, model.acc_train)
    plt.plot(model.epochs_record, model.acc_test)
    plt.legend(['train', 'test'], loc='upper left')
    plt.title("FER2013 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.plot(list(range(1, NUM_EPOCHS + 1)), model.loss_history)
    plt.title("FER2013 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")


def train_vgg16():
    """
    This is the VGG16 model, which is also mentioned in our report.
    :return: A VGG16 model in .pt format
    """
    datapath = os.path.join("data", "CK_data.h5")
    f = h5py.File(datapath, "r")
    images = np.array(f['data_samples'])
    labels = np.array(f['data_labels'])

    dataset = load_data_vgg(images, labels)
    np.random.shuffle(dataset)
    len = dataset.shape[0]
    spliter = np.split(dataset, [int(np.floor(len * 0.7)), int(len)])

    model = VGG16(alpha=0.01, epochs=25, batch_size=1, dataset=spliter, num_classes=7)
    model._train()

    dst_path = "model_data/vgg16_ck.pt"
    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))

    torch.save(model, dst_path)


def test(model_path="model_data/cnn_by_parts.pt"):
    model_path = model_path
    model = torch.load(model_path)
    model._test()


def upsampling(image, target_width, target_height):
    """
    This is a helper function used for upsampling
    :param target_width
    :param target_height
    :return: image with size (target_height, target_width)
    """
    # Require grey scale image
    dim = (target_width, target_height)
    resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    return resized


def load_data_vgg(images, label):
    n = images.shape[0]
    result = []
    for i in range(n):
        upsampled = upsampling(images[i], 224, 224)
        result.append([upsampled, label[i]])
    return np.array(result)
