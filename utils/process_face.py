from imutils import face_utils
import cv2 as cv
import dlib
import numpy as np

def detect_bipart(img, create_clahe=False, clahe=None):
    test = img

    gray = cv.cvtColor(test, cv.COLOR_BGR2GRAY)
    if create_clahe:
        gray = clahe.apply(gray)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    rects = detector(gray, 1)

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

              top_left_x = y
              top_left_y = x
            elif name == 'right_eye':
              (x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))

              bottom_left_x = y
              bottom_left_y = x + w
            elif name == 'left_eyebrow':
              (x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))

              top_right_x = y + h
              top_right_y = x + w
            elif name == 'left_eye':
              (x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))

              bottom_right_x = y + h
              bottom_right_y = x + w
            elif name == 'mouth':
              (x, y, w, h) = cv.boundingRect(np.array([shape[i:j]]))
              mouth = test[y:y + h, x:x + w]
              mouth = cv.resize(mouth, (64, 32), interpolation= cv.INTER_CUBIC)
              mouth = cv.cvtColor(mouth, cv.COLOR_BGR2GRAY)

    x = min(top_left_x, bottom_left_x)
    y = min(top_left_y, bottom_left_y)
    h = max(top_right_y - y, bottom_right_y - y)
    w = max(top_right_x - x, bottom_right_x - x)
    eyes = test[x:x+w, y:y+h]
    eyes = cv.resize(eyes, (64, 32), interpolation=cv.INTER_CUBIC)
    eyes = cv.cvtColor(eyes, cv.COLOR_BGR2GRAY)
    # mouth 100x250, eyes 50x200
    return [mouth, eyes]
