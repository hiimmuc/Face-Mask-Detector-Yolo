import time
from threading import Thread

import cv2
from facedetect_yolo import Yolov4
from imutils.video import FPS


class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


class VideoShow:
    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == 27:
                self.stopped = True

    def stop(self):
        self.stopped = True


# label = r"backup/obj.names"
# config = r"backup/yolov4-tiny-custom.cfg"
# net_path = r"backup/yolov4-tiny-custom_best.weights"

# print("[INFO] Loading net...")
# t = time.time()
# myYolo = Yolov4(net_path=net_path, config=config, label=label)
# print(f"[INFO] Done in {round(time.time() - t, 2)} s")


# def threadBoth(source=0):
#     video_getter = VideoGet(source).start()
#     video_shower = VideoShow(video_getter.frame).start()
#     fps = FPS().start()
#     delay = 0
#     while True:
#         if video_getter.stopped or video_shower.stopped:
#             video_shower.stop()
#             video_getter.stop()
#             break

#         frame = video_getter.frame
#         output_img, cond = myYolo.detector(frame, 0.3, 0.5, delay)
#         if cond:
#             delay = delay + 1 if delay <= 3 else 0
#         video_shower.frame = output_img
#         fps.update()
#     fps.stop()
#     print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#     print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# threadBoth()
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
)

class CustomDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setWindowTitle("HELLO!")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel("Something happened, is that OK?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        button = QPushButton("Press me for a dialog!")
        button.clicked.connect(self.button_clicked)
        self.setCentralWidget(button)

    def button_clicked(self, s):
        print("click", s)

        dlg = CustomDialog()  # If you pass self, the dialog will be centered over the main window as before.
        if dlg.exec_():
            print("Success!")
        else:
            print("Cancel!")

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()