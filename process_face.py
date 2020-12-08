import imutils
from imutils import face_utils
import cv2 as cv
import matplotlib.pyplot as plt
import dlib
import numpy as np

def detect_bipart(img):
    test = img

    gray = cv.cvtColor(test, cv.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    rects = detector(gray, 1)
    # fig=plt.figure(figsize=(10, 10))
    # num = 1
    # h_left_eyebrow = 0
    # x_left_eyebrow_start = 0
    # x_left_eyebrow = 0

    # h_right_eyebrow = 0
    # x_right_eyebrow = 0
    # x_right_eyebrow_start = 0
    top_left_x = 0
    top_left_y = 0
    top_right_x = 0
    top_right_y = 0
    bottom_left_x = 0
    bottom_left_y = 0
    bottom_right_x = 0
    bottom_right_y = 0
    left_eye, right_eye = None, None
    mouth = None
    for _, rect in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv.rectangle(test, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # print(name, (i, j))
            if name == 'right_eyebrow':
              (x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))
              # left_eye = test[y:y + h, x:x + w]
              # h_left_eyebrow = y
              # x_left_eyebrow = w
              # x_left_eyebrow_start = x
              top_left_x = y
              top_left_y = x

              # print('left_eyebrow')
              # print(top_left_x, top_left_y, w, h)
            elif name == 'right_eye':
              (x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))
              # left_eye = test[h_left_eyebrow:y + h, x_left_eyebrow_start:x_left_eyebrow_start + x_left_eyebrow]
              # print(h_left_eyebrow,y + h, x_left_eyebrow_start,x_left_eyebrow_start + x_left_eyebrow)
              # left_eye = imutils.resize(left_eye, width=250, inter=cv.INTER_CUBIC)
              # fig.add_subplot(3, 5, num)
              # num += 1
              # left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)
              # plt.imshow(left_eye, cmap='gray')
              bottom_left_x = y
              bottom_left_y = x + w
              # print('left_eye')
              # print(bottom_left_x, bottom_left_y, w, h)
            elif name == 'left_eyebrow':
              (x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))
              # roi = test[y:y + h, x:x + w]
              # h_right_eyebrow = y
              # x_right_eyebrow = w
              # x_right_eyebrow_start = x
              top_right_x = y + h
              top_right_y = x + w
              # print('right_eyebrow')
              # print(top_right_x, top_right_y, w, h)
            elif name == 'left_eye':
              (x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))
              # roi = test[h_right_eyebrow:y + h, x_right_eyebrow_start:x_right_eyebrow_start + x_right_eyebrow]
              # roi = test[y:y + h, x:x+w]
              # roi = imutils.resize(roi, width=250, inter=cv.INTER_CUBIC)
              # roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
              # fig.add_subplot(3, 5, num)
              # num += 1
              # plt.imshow(roi, cmap='gray')
              # plt.imshow(roi)
              bottom_right_x = y + h
              bottom_right_y = x + w
              # print('right_eye')
              # print(bottom_right_x, bottom_right_y, w, h)
            elif name == 'mouth':
              (x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))
              mouth = test[y:y + h, x:x + w]
              mouth = cv.resize(mouth, (64, 32), interpolation= cv.INTER_CUBIC)
              mouth = cv.cvtColor(mouth, cv.COLOR_BGR2GRAY)
              # plt.imshow(roi, cmap='gray')
              # plt.show()

    x = min(top_left_x, bottom_left_x)
    y = min(top_left_y, bottom_left_y)
    h = max(top_right_y - y, bottom_right_y - y)
    w = max(top_right_x - x, bottom_right_x - x)
    eyes = test[x:x+w, y:y+h]
    eyes = cv.resize(eyes, (64, 32), interpolation=cv.INTER_CUBIC)
    eyes = cv.cvtColor(eyes, cv.COLOR_BGR2GRAY)
    # mouth 100x250, eyes 50x200
    return [mouth, eyes]

# if __name__ == "__main__":
#     image = cv.imread('portrait.jpeg')
#     mouth, eyes = detect_bipart(image)
#     plt.imshow(eyes, cmap='gray')
#     plt.show()