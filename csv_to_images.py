import os
import csv
import numpy as np
import imageio

csv_file = "datasets/fer2013/fer2013.csv"
OUTPUT = "datasets/fer2013/images"

if __name__ == "__main__":
    w, h = 48, 48
    image = np.zeros((h, w), dtype=np.uint8)
    id = 1
    with open(csv_file, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        headers = next(datareader)
        print(headers)
        for row in datareader:
            emotion = row[0]
            pixels = list(map(int, row[1].split()))
            usage = row[2]

            pixels_array = np.asarray(pixels)

            image = pixels_array.reshape(w, h)
            image_folder = os.path.join(OUTPUT, emotion)
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            image_file = os.path.join(image_folder, str(id) + '.jpg')
            imageio.imwrite (image_file, image)
            id += 1
            if id % 100 == 0:
                print('Processed {} images'.format(id))

    print("Finished processing {} images".format(id))