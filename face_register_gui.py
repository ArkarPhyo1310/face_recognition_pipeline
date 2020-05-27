import PIL
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import cv2
import datetime
import os
import numpy as np
from facenet_pytorch import MTCNN


class FaceRegister(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.width, self.height = 640, 480
        self.frame = None
        self.count = 0
        self.dataset_path = 'dataset/'
        self.bboxes = None

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # load the mtccnn for face detection
        self.face_detector = MTCNN(select_largest=False,
                              device='cuda:0',
                              margin=20,
                              thresholds=[0.6, 0.7, 0.8],
                              image_size=224)

        # creating elements
        self.text = tk.Label(self, text='Enter Your Full Name: ')
        self.name = tk.Entry(self, width=50)
        self.name.var = tk.StringVar(self)
        self.name['textvariable'] = self.name.var
        self.name.var.trace_add('write', self.toggle_state)
        self.enter = tk.Button(self, text="Enter", command=self.show_text, state='disabled')

        self.webcam_frame = tk.Frame(self, width=640, height=480)
        self.lmain = tk.Label(master=self.webcam_frame)
        self.welcome_text = tk.Label(master=self.webcam_frame, text="(Enter your name with space..)")
        self.save_btn = tk.Button(master=self.webcam_frame, text="Close", command=self.close_window)
        self.output_text = tk.Label(master=self.webcam_frame)

        # creating grid structure
        self.text.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.name.grid(row=0, column=1, padx=10, pady=10, columnspan=3)
        self.enter.grid(row=0, column=4, padx=10, pady=10, sticky='e')

        self.webcam_frame.grid(row=1, column=0, padx=10, pady=5, rowspan=5, columnspan=5)
        self.welcome_text.grid(row=0, column=0, pady=5)
        self.lmain.grid(row=1, column=0, pady=5)
        self.save_btn.grid(row=2, column=0, padx=10, pady=5)
        self.output_text.grid(row=3, column=0, padx=10, pady=5)

    def toggle_state(self, *_):
        if len(self.name.get()) > 7 and ' ' in self.name.get():
            self.enter['state'] = 'normal'
        else:
            self.enter['state'] = 'disabled'

    def show_text(self):
        if os.path.exists(self.dataset_path + self.name.get().replace(" ", "_")):
            self.welcome_text.configure(text="You have been registered! " + self.name.get().upper(), font=("Arial", 15))
        else:
            self.welcome_text.configure(text="Welcome! " + self.name.get().upper(), font=("Arial", 20))
            self.output = self.dataset_path + self.name.get().replace(" ", "_")
            os.mkdir(self.dataset_path + self.name.get().replace(" ", "_"))
            self.save_btn.configure(text="Save Face", command=self.save_face)
            self.show_webcam()

    def show_webcam(self):
        _, self.frame = self.cap.read()
        self.frame = cv2.flip(self.frame, 1)
        cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(cv2image)
        # extract bboxes of faces in an image
        self.bboxes, _ = self.face_detector.detect(img)

        # Draw faces
        frame_draw = img.copy()
        draw = ImageDraw.Draw(frame_draw)
        if self.bboxes is not None:
            for bbox in self.bboxes:
                draw.rectangle(bbox.tolist(), outline=(255, 0, 0), width=4)

        imgtkinter = ImageTk.PhotoImage(image=frame_draw)
        self.lmain.imgtk = imgtkinter
        self.lmain.configure(image=imgtkinter)
        self.lmain.after(10, self.show_webcam)

    def save_face(self):
        ts = datetime.datetime.now()
        if self.bboxes is not None:
            filename = "{}.png".format(ts.strftime("%Y%m%d-%H%M%S"))
            path = os.path.sep.join((self.output, filename))

            # save the file
            cv2.imwrite(path, self.frame.copy())
            self.count += 1
            self.output_text.configure(text="{} image(s) created...".format(self.count))
        else:
            self.output_text.configure(text="face not found...")

    def close_window(self):
        self.destroy()


def main():
    app = FaceRegister()
    app.title("Face Registration System")
    app.resizable(False, False)
    app.mainloop()


if __name__ == '__main__':
    main()
