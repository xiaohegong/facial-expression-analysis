"""
This file includes several ways to read the data samples from CK+/Fer2013 and then save desired features into .h5 format.
    1. Normal: Save images with their corresponding labels as two dataset into data file
    2. Bipart: Preprocess images and save the eyes and mouth parts respectively into two dataset.
    3. Hog: Calculate hog feature and create a dataset to store HOG.
"""

import os
import cv2 as cv
import numpy as np
import h5py
import pandas as pd
from utils.process_face import detect_bipart

FER13_BY_PARTS_DATA = "data/fer13_bipart_data.h5"
FER13_DATA = "fer2013_data.h5"
FER13_DATASET = 'datasets/fer2013'
CK_DATASET = 'datasets/CK+48'
IMAGE_LOCATION = 'tmp/image.png'

anger = os.path.join(CK_DATASET, 'anger')
contempt = os.path.join(CK_DATASET, 'contempt')
disgust = os.path.join(CK_DATASET, 'disgust')
fear = os.path.join(CK_DATASET, 'fear')
happy = os.path.join(CK_DATASET, 'happy')
sadness = os.path.join(CK_DATASET, 'sadness')
surprise = os.path.join(CK_DATASET, 'surprise')


def save_fer2013():
    """
    This is normal save -- just store images and labels from FER2013
    :return: Two datasets -- One for images and one for labels.
             labels[i] is the true label of images[i]

    0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    """
    data = readcsv("./datasets/fer2013/fer2013.csv")
    samples, labels = process_data(data)
    datapath = os.path.join('data', 'fer2013_data.h5')
    datafile = h5py.File(datapath, 'w')
    datafile.create_dataset("data_samples", dtype='float32', data=samples)
    datafile.create_dataset("data_labels", dtype='int32', data=labels)
    datafile.close()


def save_ck_plus():
    """
    This is normal save -- just store images and labels from CK+
    :return: Two datasets -- One for images and one for labels.
             labels[i] is the true label of images[i]

    0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    """
    samples, labels = [], []

    datapath = os.path.join('data', 'CK_data.h5')

    if not os.path.exists(os.path.dirname(datapath)):
        os.makedirs(os.path.dirname(datapath))

    folders = [anger, contempt, disgust, fear, happy, sadness, surprise]
    for i, folder in enumerate(folders):
        images = np.array(os.listdir(folder))
        np.random.shuffle(images)  # shuffle image
        for image in images:
            image = cv.imread(os.path.join(folder, image), cv.IMREAD_GRAYSCALE)
            samples.append(image)
            labels.append(i)

    # Create datasets
    datafile = h5py.File(datapath, 'w')
    datafile.create_dataset("data_samples", dtype='float32', data=samples)
    datafile.create_dataset("data_labels", dtype='int32', data=labels)
    datafile.close()


def save_hog_bipart():
    """
    Store eyes and mouth parts as well as the hog of them for CK+.
    :return: 5 datasets -- Eyes, mouth, eyes_hog, mouth_hog, labels

    0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    """

    eyes_samples = []
    mouth_samples = []
    hogs_eyes = []
    hogs_mouth = []
    labels = []

    datapath = os.path.join('data', 'CK_bipart_hog.h5')

    if not os.path.exists(os.path.dirname(datapath)):
        os.makedirs(os.path.dirname(datapath))

    folders = [anger, contempt, disgust, fear, happy, sadness, surprise]
    # Loop through all folders
    for i, folder in enumerate(folders):
        images = np.array(os.listdir(folder))
        np.random.shuffle(images)  # shuffle image

        for image in images:
            image = cv.imread(os.path.join(folder, image))
            mouth, eyes, hog_eyes, hog_mouth = detect_bipart(image)
            eyes_samples.append(eyes)
            mouth_samples.append(mouth)
            hogs_eyes.append(hog_eyes)
            hogs_mouth.append(hog_mouth)
            labels.append(i)

    # Create dataset
    datafile = h5py.File(datapath, 'w')
    datafile.create_dataset("eyes_samples", dtype='float32', data=eyes_samples)
    datafile.create_dataset("mouth_samples", dtype='float32', data=mouth_samples)
    datafile.create_dataset("hog_eyes", dtype='float32', data=hogs_eyes)
    datafile.create_dataset("hog_mouth", dtype='float32', data=hogs_mouth)
    datafile.create_dataset("data_labels", dtype='int32', data=labels)
    datafile.close()


def save_fer2013_bipart():
    """
    Store eyes and mouth parts for Fer2013.csv
    :return: 3 datasets -- Eyes, mouth, labels

    0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    """
    undetected, processed = 0, 0
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    eyes_samples = []
    mouth_samples = []
    labels_new = []

    datapath = os.path.join('data', FER13_DATA)
    f = h5py.File(datapath, "r")
    images = np.array(f['data_samples'])
    labels = np.array(f['data_labels'])

    for i in range(images.shape[0]):
        imageio.imwrite(IMAGE_LOCATION, images[i])
        image = cv.imread(IMAGE_LOCATION)
        # Detect bipart features
        try:
            eyes, mouth = detect_bipart(image, create_clahe=True, clahe=clahe)
            processed += 1
            if processed % 50 == 0:
                print("Processed {} images, undetected {}".format(processed, undetected))
        except cv.error as e:
            undetected += 1
            continue

        eyes_samples.append(eyes)
        mouth_samples.append(mouth)
        labels_new.append(labels[i])
    # Create datasets
    datafile = h5py.File(FER13_BY_PARTS_DATA, 'w')
    datafile.create_dataset("eyes_samples", dtype='float32', data=eyes_samples)
    datafile.create_dataset("mouth_samples", dtype='float32', data=mouth_samples)
    datafile.create_dataset("data_labels", dtype='int32', data=labels_new)
    datafile.close()


def readcsv(filePath):
    data = pd.read_csv(filePath)
    print(data.shape)
    return data


def process_data(data):
    samples = []
    labels = []
    for _, row in data.iterrows():
        pixels = row["pixels"].split(" ")
        pixel_len = len(pixels)
        side = np.sqrt(pixel_len).astype(np.int)
        for i in range(pixel_len):
            pixels[i] = int(pixels[i])
        pixels = np.array(pixels).reshape((side, side))
        samples.append(pixels)
        labels.append(row["emotion"])

    return np.array(samples), np.array(labels)
