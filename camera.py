from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk
import time
import numpy as np

class FERApp:
    def __init__(self, vid_src=0):
        self.name = "FER converter"
        self.vid_src = vid_src
        self.add_widgets()
        self.update()
        self.window.mainloop()

    def add_widgets(self):
        self.window = Tk()
        self.window.geometry('1200x800')
        self.window.title(self.name)
        self.window.resizable(0, 0) # set it to non-resizable
        # self.window.wm_iconbitmap("cam.ico")
        self.window['bg'] = "black"
        self.video = CameraCapture(self.vid_src)
        Label(self.window, text="Welcome to our FER converter", font=15,
                            bg="black", fg= "white").pack(side=TOP, fill=BOTH)
        # create canvas for video
        self.canvas = Canvas(self.window, width = self.video.width, height=self.video.height, bg="white")
        self.canvas.pack(side="top", anchor=NW)
        
        self.snapshot_btn = Button(self.window, text="take photo", width=100, bg="red",
                            activebackground="red", command=self.snapshot)
        self.snapshot_btn.place(x=0, y=600)
        # add emotion labels
        self.add_emotion_labels()

    def add_emotion_labels(self):
        score = np.array([0.777, 0.625, 0.324, 0.2345, 0.113, 0.55343, 3.9234])
        score = score / score.sum()
        emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
        for i in range(len(emotions)):
            label = Label(self.window, text=emotions[i] +":"+ str(round(score[i], 6)), font=15,
                                bg="white", fg= "black")
            label.place(x=800, y=100+50*i)

    def snapshot(self):
        # used to capture the image with emoji, TODO
        captured, frame = self.video.getFrame()
        if captured:
            image = "IMG-" + time.strftime("%H-%M-%S-%d-%m") + ".jpg"
            cv.imwrite(image, cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            Label(self.window, text="image saved").place(x=430, y=510)
    
    def update(self):
        camera_open, frame = self.video.getFrame()

        if camera_open:
            self.cur_frame = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.cur_frame, anchor=NW)
        
        self.window.after(15, self.update) # rate to call update function




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
    FERApp()