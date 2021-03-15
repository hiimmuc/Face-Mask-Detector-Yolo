import os
import sys

import cv2
import numpy as np
from facedetect_yolo import Yolov4
from GUI import Ui_Phanmemtheodoikhautrang
from imutils.video import FPS
from playsound import playsound
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, list)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.data = Data()
        self.model = self.data.model
        self.delay = 0

    def run(self):
        # capture from web cam
        print("[INFO] Start recording...")
        cap = cv2.VideoCapture(0)
        # stream = VideoStream().start()
        self.fps = FPS().start()
        while self._run_flag:
            ret, frame = cap.read()
            # frame = stream.read()
            # ret = True
            if ret:
                output_img, self.value = self.model.detector(frame, 0.4, 0.6, self.delay)
                if not self.value[0]:
                    self.delay = self.delay + 1 if self.delay < 3 else 0
                self.change_pixmap_signal.emit(output_img, self.value)
                self.fps.update()
        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class Data(object):
    def __init__(self):
        self.config = r"backup/yolov4-tiny-custom.cfg"
        self.net_path = r"backup/yolov4-tiny-custom_best.weights"
        self.label = r"backup/obj.names"
        self.model = Yolov4(net_path=self.net_path, config=self.config, label=self.label)


class CustomDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setWindowTitle("Nhắc nhở")
        self.setWindowIcon(QtGui.QIcon("Photos/1.ico"))

        QBtn = QDialogButtonBox.Ok

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()
        message = QLabel("Hãy đeo khẩu trang vào \nĐeo khẩu trang là hành động bảo vệ bản thân, gia đình và xã hội")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class App(Ui_Phanmemtheodoikhautrang, VideoThread):
    def __init__(self, MainWindow) -> None:
        super().__init__()
        self.setupUi(MainWindow)
        self.delay = 0
        self.notice = "GOOD!"
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_all)
        # start the thread
        self.button.clicked.connect(self.thread.start)

    @pyqtSlot(np.ndarray, list)
    def update_all(self, cvimg, value=[0, 0, 0]):
        """Updates the image_label with a new opencv image"""
        _translate = QtCore.QCoreApplication.translate
        qtimg = self.convert_cv_qt(cvimg)
        nums, masked, unmasked = value
        incorrectly_masked = nums - masked - unmasked
        incorrectly_masked = incorrectly_masked if incorrectly_masked > 0 else 0
        masked = nums - incorrectly_masked - unmasked
        Noti = ""
        if unmasked >= 1 and masked == 0:
            self.notice = "Đeo khẩu trang vào"
        elif unmasked == 0:
            self.notice = "GOOD!"
        Noti = str(self.notice)
        self.noti.setText(_translate("MainWindow", f"<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:600;\">Thông báo:</span></p><p align=\"justify\"><br/><span style=\" font-size:9pt;\">Phát hiện: {nums}</span></p><p align=\"justify\"><span style=\" font-size:9pt;\">Đã đeo khẩu trang: {masked}</span></p><p align=\"justify\"><span style=\" font-size:9pt;\">Đeo chưa đúng: {incorrectly_masked}</span></p><p align=\"justify\"><span style=\" font-size:9pt;\">Chưa đeo khẩu trang: {unmasked}</span></p></body></html><p align=\"justify\"><br/><span style=\" font-size:9pt;\">" + Noti + "</span></p>"))
        self.screen.setPixmap(qtimg)

    def convert_cv_qt(self, cvimg):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    @pyqtSlot()
    def popup_noti(self):
        # path = os.path.abspath("backup/voice.wav")
        # playsound(path)
        # dlg = CustomDialog()
        # dlg.exec_()
        pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = App(MainWindow=MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
