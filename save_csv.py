import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

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



if __name__ == "__main__":
    data = readcsv("./datasets/fer2013/fer2013.csv")
    samples, labels = process_data(data)
    datapath = os.path.join('data','fer2013_data.h5')
    datafile = h5py.File(datapath, 'w')
    datafile.create_dataset("data_samples", dtype = 'float32', data=samples)
    datafile.create_dataset("data_labels", dtype = 'int32', data=labels)
    datafile.close()
