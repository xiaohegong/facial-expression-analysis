"""
file used to save ck+ data
"""
import os
import cv2 as cv
import numpy as np
import h5py
import imageio
from process_face import detect_bipart

# label 0: anger
# label 1: contempt
# label 2: disgust
# label 3: fear
# label 4: happy
# label 5: sadness
# label 6: surprise

FER13_BY_PARTS_DATA = "data/fer13_bipart_data.h5"
FER13_DATA = "fer2013_data.h5"
FER13_DATASET = 'datasets/fer2013'
IMAGE_LOCATION = 'tmp/image.png'

if __name__ == "__main__":
    undetected, processed = 0, 0
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # used to store images and its coresponding labels
    #samples = []
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

    print(np.shape(eyes_samples))
    print(np.shape(labels_new))


    datafile = h5py.File(FER13_BY_PARTS_DATA, 'w')
    datafile.create_dataset("eyes_samples", dtype = 'float32', data=eyes_samples)
    datafile.create_dataset("mouth_samples", dtype = 'float32', data=mouth_samples)
    datafile.create_dataset("data_labels", dtype = 'int32', data=labels_new)
    datafile.close()