"""
file used to save ck+ data
"""
import os
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
import h5py
from process_face import detect_bipart

# label 0: anger
# label 1: disgust
# label 2: fear
# label 3: happy
# label 4: sadness
# label 5: surprise
# label 6: neutral

FER13_BY_PARTS_DATA = "data/fer13_bipart_data_tmp.h5"
FER13_DATA = "fer2013_data.h5"
FER13_DATASET = 'datasets/fer2013'
FER13_IMAGES = 'datasets/fer2013/images'



if __name__ == "__main__":
    undetected, processed = 0, 0
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    anger = os.path.join(FER13_IMAGES, '0')
    neutral = os.path.join(FER13_IMAGES, '6')
    disgust = os.path.join(FER13_IMAGES, '1')
    fear = os.path.join(FER13_IMAGES, '2')
    happy = os.path.join(FER13_IMAGES, '3')
    sadness = os.path.join(FER13_IMAGES, '4')
    surprise = os.path.join(FER13_IMAGES, '5')


    # used to store images and its coresponding labels
    eyes_samples = []
    mouth_samples = []
    labels = []

    datapath = os.path.join('data','CK_bipart_data.h5')

    if not os.path.exists(os.path.dirname(datapath)):
        os.makedirs(os.path.dirname(datapath))

    folders = [anger, neutral, disgust, fear, happy, sadness, surprise]

    for i, folder in enumerate(folders):
        images = np.array(os.listdir(folder))
        print("Reading folder " + folder)
        for image in images:
            image = cv.imread(os.path.join(folder,image))

            try:
                eyes, mouth = detect_bipart(image, create_clahe=True)
                processed += 1
                if processed % 100 == 0:
                    print("Processed {} images".format(processed))
            except cv.error as e:
                undetected += 1
                print(e)
                continue
            eyes_samples.append(eyes)
            mouth_samples.append(mouth)
            labels.append(i)

    print(np.shape(eyes_samples))
    print(np.shape(labels))
    print("{} images undetected eyes".format(undetected))

    datafile = h5py.File(FER13_BY_PARTS_DATA, 'w')
    datafile.create_dataset("eyes_samples", dtype = 'float32', data=eyes_samples)
    datafile.create_dataset("mouth_samples", dtype = 'float32', data=mouth_samples)
    datafile.create_dataset("data_labels", dtype = 'int32', data=labels)
    datafile.close()