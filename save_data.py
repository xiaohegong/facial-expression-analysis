"""
file used to save ck+ data
"""
import os
import cv2 as cv
import numpy as np
import h5py

# label 0: anger
# label 1: contempt
# label 2: disgust
# label 3: fear
# label 4: happy
# label 5: sadness
# label 6: surprise


ck = 'datasets\CK+48'
anger = os.path.join(ck, 'anger')
contempt = os.path.join(ck, 'contempt')
disgust = os.path.join(ck, 'disgust')
fear = os.path.join(ck, 'fear')
happy = os.path.join(ck, 'happy')
sadness = os.path.join(ck, 'sadness')
surprise = os.path.join(ck, 'surprise')


# used to store images and its coresponding labels
samples = []
labels = []

datapath = os.path.join('data','CK_data.h5')

if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

folders = [anger, contempt, disgust, fear, happy, sadness, surprise]

for i, folder in enumerate(folders):
    images = os.listdir(folder)
    for image in images:
        image = cv.imread(os.path.join(folder,image), cv.IMREAD_GRAYSCALE)
        samples.append(image)
        labels.append(i)

print(np.shape(samples))
print(np.shape(labels))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("data_samples", dtype = 'float32', data=samples)
datafile.create_dataset("data_labels", dtype = 'int32', data=labels)
datafile.close()