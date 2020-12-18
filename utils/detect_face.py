import imutils
from imutils import face_utils
import cv2 as cv
import matplotlib.pyplot as plt
import dlib
import numpy as np
import torch

emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def setup():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("utils/pretrained_face_detector/shape_predictor_68_face_landmarks.dat")
    return detector, predictor


def detect_face(image, model, detector, predictor):
    image = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    predictions = []
    for _, rect in enumerate(rects):  # for each face detected
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        # predict emotion
        resized_image = cv.resize(gray[y:y + h, x:x + w], (224, 224), cv.INTER_AREA)
        prediction, emotion_idx = model.predict(resized_image)
        prediction = prediction.cpu().detach().numpy()  # convert it to numpy array
        predictions.append(prediction[0])
        # print("LOG")
        # print("prediction", prediction.cpu().detach().numpy())
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw rectangle in detected face
        cv.putText(image, emotions[emotion_idx], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return image, predictions


if __name__ == "__main__":
    image = cv.imread("test_images/test.jpg")
    model_path = "model_data/vgg16.pt"
    model = torch.load(model_path)
    detector, predictor = setup()
    image, predictions = detect_face(image, model, detector, predictor)  # detect face
