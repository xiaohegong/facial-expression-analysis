from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk
import time
import torch
from detect_face import detect_face, setup
import argparse
from model.dcnn_model import CustomizedCNNModel

from model.vgg16 import VGG16


class FERApp:
    def __init__(self, model, vid_src=0):
        self.counter = 0
        self.detector, self.predictor = setup()
        self.counter = 0
        self.model = model
        self.name = "FER converter"
        self.vid_src = vid_src
        self.add_widgets()
        self.update()
        self.window.mainloop()

    def add_widgets(self):
        self.window = Tk()
        self.window.geometry('1200x800')
        self.window.title(self.name)
        self.window.resizable(0, 0)  # set it to non-resizable
        # self.window.wm_iconbitmap("cam.ico")
        self.window['bg'] = "black"
        self.video = CameraCapture(self.vid_src)
        Label(self.window, text="Welcome to our FER converter", font=15,
              bg="black", fg="white").pack(side=TOP, fill=BOTH)
        # create canvas for video
        self.canvas = Canvas(self.window, width=self.video.width, height=self.video.height, bg="white")
        self.canvas.pack(side="top", anchor=NW)

        self.snapshot_btn = Button(self.window, text="take photo", width=100, bg="red",
                                   activebackground="red", command=self.snapshot)
        self.snapshot_btn.place(x=0, y=600)
        # add emotion labels
        self.add_emotion_labels()

    def add_emotion_labels(self):
        self.angerLabel = Label(self.window, text="anger:0.00", font=15,
                                bg="white", fg="black")
        self.angerLabel.place(x=800, y=100)
        self.contemptLabel = Label(self.window, text="contempt:0.00", font=15,
                                   bg="white", fg="black")
        self.contemptLabel.place(x=800, y=150)
        self.disgustLabel = Label(self.window, text="disgust:0.00", font=15,
                                  bg="white", fg="black")
        self.disgustLabel.place(x=800, y=200)
        self.fearLabel = Label(self.window, text="fear:0.00", font=15,
                               bg="white", fg="black")
        self.fearLabel.place(x=800, y=250)
        self.happyLabel = Label(self.window, text="happy:0.00", font=15,
                                bg="white", fg="black")
        self.happyLabel.place(x=800, y=300)
        self.sadnessLabel = Label(self.window, text="sadness:0.00", font=15,
                                  bg="white", fg="black")
        self.sadnessLabel.place(x=800, y=350)
        self.surpriseLabel = Label(self.window, text="surprise:0.00", font=15,
                                   bg="white", fg="black")
        self.surpriseLabel.place(x=800, y=400)

    def snapshot(self):
        # used to capture the image with emoji, TODO
        captured, frame = self.video.getFrame()
        frame, predictions = detect_face(frame, self.model, self.detector, self.predictor)
        if captured:
            image = "IMG-" + time.strftime("%H-%M-%S-%d-%m") + ".jpg"
            cv.imwrite(image, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            Label(self.window, text="image saved").place(x=430, y=510)

    def get_stats(self):
        camera_open, frame = self.video.getFrame()
        image, predictions = detect_face(frame, self.model, self.detector, self.predictor)
        if len(predictions) > 0:
            emotion = predictions[0]
            # FER2013: 0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral
            self.angerLabel["text"] = "angry:" + str(emotion[0])
            self.disgustLabel["text"] = "disgust:" + str(emotion[1])
            self.fearLabel["text"] = "fear:" + str(emotion[2])
            self.happyLabel["text"] = "happy:" + str(emotion[3])
            self.sadnessLabel["text"] = "sad:" + str(emotion[4])
            self.surpriseLabel["text"] = "surprise:" + str(emotion[5])
            self.contemptLabel["text"] = "neutral:" + str(emotion[6])

    def update(self):
        self.counter += 1
        camera_open, frame = self.video.getFrame()
        # process image here
        if self.counter % 10 == 0:
            self.counter = 0
            image, predictions = detect_face(frame, self.model, self.detector, self.predictor)
            if len(predictions) > 0:
                emotion = predictions[0]
                # FER2013: 0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral
                self.angerLabel["text"] = "angry:" + str(round(emotion[0], 3))
                self.disgustLabel["text"] = "disgust:" + str(round(emotion[1], 3))
                self.fearLabel["text"] = "fear:" + str(round(emotion[2], 3))
                self.happyLabel["text"] = "happy:" + str(round(emotion[3], 3))
                self.sadnessLabel["text"] = "sad:" + str(round(emotion[4], 3))
                self.surpriseLabel["text"] = "surprise:" + str(round(emotion[5], 3))
                self.contemptLabel["text"] = "neutral:" + str(round(emotion[6], 3))

        if camera_open:
            self.cur_frame = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.cur_frame, anchor=NW)

        self.window.after(30, self.update)  # rate to call update function


# camera class
class CameraCapture:
    def __init__(self, video_source=0):
        self.video = cv.VideoCapture(video_source)
        if not self.video.isOpened():
            raise ValueError("Unable to open this camera")
        self.width = self.video.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv.CAP_PROP_FRAME_HEIGHT)

    def getFrame(self):
        if self.video.isOpened():
            isTrue, frame = self.video.read()
            if isTrue:
                return (isTrue, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        return (isTrue, None)

    def __del__(self):
        # close camera
        if self.video.isOpened():
            self.video.release()


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Facial Expression Recognition')
    parser.add_argument('--model', default='model_data/cnn_fer2013.pt')
    arg = parser.parse_args()

    model_path = arg.model
    model = torch.load(model_path, map_location=device)
    FERApp(model)
