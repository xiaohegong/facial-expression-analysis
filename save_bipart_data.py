"""
file used to save ck+ data
"""
import os
import cv2 as cv
import numpy as np
import h5py
from process_face import detect_bipart

# label 0: anger
# label 1: contempt
# label 2: disgust
# label 3: fear
# label 4: happy
# label 5: sadness
# label 6: surprise

if __name__ == "__main__":
    ck = 'datasets/CK+48'
    anger = os.path.join(ck, 'anger')
    contempt = os.path.join(ck, 'contempt')
    disgust = os.path.join(ck, 'disgust')
    fear = os.path.join(ck, 'fear')
    happy = os.path.join(ck, 'happy')
    sadness = os.path.join(ck, 'sadness')
    surprise = os.path.join(ck, 'surprise')


    # used to store images and its coresponding labels
    #samples = []
    eyes_samples = []
    mouth_samples = []
    hogs_eyes = []
    hogs_mouth = []
    labels = []

    datapath = os.path.join('data','CK_bipart_hog.h5')

    if not os.path.exists(os.path.dirname(datapath)):
        os.makedirs(os.path.dirname(datapath))

    folders = [anger, contempt, disgust, fear, happy, sadness, surprise]

    for i, folder in enumerate(folders):
        images = np.array(os.listdir(folder))
        np.random.shuffle(images) # shuffle image
        for image in images:
            image = cv.imread(os.path.join(folder,image))
            # samples.append(image)
            mouth, eyes, hog_eyes, hog_mouth = detect_bipart(image)
            eyes_samples.append(eyes)
            mouth_samples.append(mouth)
            hogs_eyes.append(hog_eyes)
            hogs_mouth.append(hog_mouth)
            labels.append(i)

    print(np.shape(hogs_eyes))
    print(np.shape(hogs_mouth))

    datafile = h5py.File(datapath, 'w')
    datafile.create_dataset("eyes_samples", dtype = 'float32', data=eyes_samples)
    datafile.create_dataset("mouth_samples", dtype = 'float32', data=mouth_samples)
    datafile.create_dataset("hog_eyes", dtype = 'float32', data=hogs_eyes)
    datafile.create_dataset("hog_mouth", dtype = 'float32', data=hogs_mouth)
    datafile.create_dataset("data_labels", dtype = 'int32', data=labels)
    datafile.close()