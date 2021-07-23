import os
import cv2
import sys
from sqlite3 import connect
from PIL import Image
import numpy as np
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QThread, Signal
from PySide6.QtCore import *
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QDialog, QTableWidgetItem, QVBoxLayout
from PySide6.QtUiTools import QUiLoader


def convertCvImage2QtImage(cv_image):
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(rgb_image).convert('RGB')
    return QPixmap.fromImage(ImageQt(PIL_image)).scaled(480, 360, Qt.KeepAspectRatio)


class Camera(QThread):
    frame_signal = Signal(object, object)

    def __init__(self):
        super(Camera, self).__init__()
        # self.frame_signal = Signal()

    def make_masks(self, image):
        # w, h = image.shape

        # mask 1
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray_inv = cv2.GaussianBlur(255 - image_gray, (21, 21), 0)
        out_1 = cv2.divide(image_gray, 255 - image_gray_inv, scale=256.0)
        out_1 = cv2.resize(out_1, (0, 0), fx=0.5, fy=0.5)  # half size
        out_1 = cv2.cvtColor(out_1, cv2.COLOR_GRAY2BGR)

        # mask 2
        image_equal = cv2.equalizeHist(image_gray)
        image_equal = cv2.resize(image_equal, (0, 0), fx=0.5, fy=0.5)  # half size
        image_equal = cv2.cvtColor(image_equal, cv2.COLOR_GRAY2BGR)

        # mask 3
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (0, 0), fx=0.5, fy=0.5)

        # mask 4
        image_bt_not = cv2.bitwise_not(image)
        image_bt_not = cv2.resize(image_bt_not, (0, 0), fx=0.5, fy=0.5)

        # mask 5
        image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        overlay = np.full((image.shape[0], image.shape[1], 4), (20,66,112,1), dtype='uint8')
        cv2.addWeighted(overlay, 0.6, image_2, 1.0, 0, image_2)
        image_sepia = cv2.cvtColor(image_2, cv2.COLOR_BGRA2BGR)
        image_sepia = cv2.resize(image_sepia, (0, 0), fx=0.5, fy=0.5)

        # mask 6
        half_gray_1 = image_gray[:round(image.shape[0]/2), :]
        half_gray_2 = image_gray[round(image.shape[0] / 2):, :]
        _, mask1 = cv2.threshold(half_gray_1, 90, 255, cv2.THRESH_TOZERO)
        _, mask2 = cv2.threshold(half_gray_2, 120, 255, cv2.THRESH_TRUNC)
        half_binary = cv2.vconcat([mask1, mask2])
        half_binary = cv2.cvtColor(half_binary, cv2.COLOR_GRAY2BGR)
        half_binary = cv2.resize(half_binary, (0, 0), fx=0.5, fy=0.5)

        # mask 7
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        image_h = cv2.merge((h+50, s, v))
        image_h = cv2.cvtColor(image_h, cv2.COLOR_HSV2BGR)
        image_h = cv2.resize(image_h, (0, 0), fx=0.5, fy=0.5)

        # mask 8
        part_1 = image[:round(image.shape[0]/3), :, :]
        part_1 = cv2.bitwise_not(part_1)
        part_2 = image[round(image.shape[0]/3):round(image.shape[0]/1.5), :, :]
        part_3 = image[round(image.shape[0] / 1.5):, :, :]
        part_3 = cv2.bitwise_not(part_3)

        three_part = cv2.vconcat([part_1, part_2, part_3])
        three_part = cv2.resize(three_part, (0, 0), fx=0.5, fy=0.5)

        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        # print(half_binary.shape, image_h.shape, three_part.shape)
        return cv2.vconcat([
            cv2.hconcat([image, out_1, image_bt_not]),
            cv2.hconcat([image_equal, image_rgb, image_sepia]),
            cv2.hconcat([half_binary, image_h, three_part])
        ])

    # ----------------------------------------------------------------------------
    def save_pic(self, event, x, y, flags, param):
        # print(x, y, self.frame_masked.shape[0], self.frame_masked.shape[1])
        if event == cv2.EVENT_LBUTTONDOWN:
            img = np.copy(self.frame_masked)
            if y < (img.shape[0] / 3):
                if x < img.shape[1] / 3:
                    img = img[:round(img.shape[0]/3), :round(img.shape[1]/3), :]
                elif img.shape[1] / 3 < x < img.shape[1] / 1.5:
                    img = img[:round(img.shape[0]/3), round(img.shape[1]/3):round(img.shape[1]/1.5), :]
                elif img.shape[1] / 1.5 < x:
                    img = img[:round(img.shape[0]/3), round(img.shape[1]/1.5):, :]

            elif (img.shape[0] / 3) < y < (img.shape[0] / 1.5):
                if x < img.shape[1] / 3:
                    img = img[round(img.shape[0]/3):round(img.shape[0]/1.5), :round(img.shape[1]/3), :]
                elif img.shape[1] / 3 < x < img.shape[1] / 1.5:
                    img = img[round(img.shape[0]/3):round(img.shape[0]/1.5), round(img.shape[1]/3):round(img.shape[1]/1.5), :]
                elif img.shape[1] / 1.5 < x:
                    img = img[round(img.shape[0]/3):round(img.shape[0]/1.5), round(img.shape[1]/1.5):, :]

            elif (img.shape[0] / 1.5) < y:
                if x < img.shape[1] / 3:
                    img = img[round(img.shape[0]/1.5):, :round(img.shape[1]/3), :]
                elif img.shape[1] / 3 < x < img.shape[1] / 1.5:
                    img = img[round(img.shape[0]/1.5):, round(img.shape[1]/3):round(img.shape[1]/1.5), :]
                elif img.shape[1] / 1.5 < x:
                    img = img[round(img.shape[0]/1.5):, round(img.shape[1]/1.5):, :]

            cv2.imwrite('employee_masked.jpg', img)
            orig = self.frame_masked[:round(self.frame_masked.shape[0]/3), :round(self.frame_masked.shape[1]/3), :]
            cv2.imwrite('employee_original.jpg', orig)

    # ----------------------------------------------------------------------------
    def run(self):
        video = cv2.VideoCapture(0)
        while True:
            flag, self.frame = video.read()

            # Wait for 'q' key to stop the program
            if cv2.waitKey(1) == ord('q'):
                break

            self.frame_masked = self.make_masks(self.frame)

            if flag:
                # cv2.imshow('pic', self.frame_masked)
                self.frame_signal.emit(self.frame, self.frame_masked)
                # cv2.setMouseCallback('pic', self.save_pic)
                # window.ui.stackedWidget.setCurrentWidget(window.ui.blue)
                # self.frame = convertCvImage2QtImage(self.frame)
                # window.ui.video_main.setPixmap(self.frame)


# ****************************************************************************************
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        loader = QUiLoader()
        self.ui = loader.load('stacked.ui')

        self.ui.show()

        self.ui.stackedWidget.setCurrentWidget(self.ui.home)
        self.ui.blue_btn.clicked.connect(self.show_blue)
        self.ui.red_btn.clicked.connect(self.show_red)
        self.ui.yellow_btn.clicked.connect(self.show_yellow)
        self.ui.menu_btn.clicked.connect(self.back_to_home)

        self.webcam = Camera()
        self.picture_signal = Signal()

    def show_red(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.red)
        self.ui.camera_btn.clicked.connect(self.camera)
        # self.ui.take_photo.clicked.connect(self.record_picture)
        self.ui.submit_btn.clicked.connect(self.submit_employee)

    def submit_employee(self):
        if self.ui.fname.text() == '' or self.ui.lname.text() == '' or self.ui.code.text() == '':
            self.ui.lbl_error.setText('please fill all fields')
        elif not os.path.exists('employee_masked.jpg'):
            self.ui.lbl_error.setText('please take a picture')
        else:
            self.ui.lbl_error.setText('ok, please wait...')

            # check database (not duplicated information)
            data_in_database = self.db_fetch()
            duplicate = any(self.ui.code.text() in item for item in data_in_database)

            if duplicate:
                self.ui.lbl_error.setText('this person submitted before.')
            else:
                # face detection
                face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                img = cv2.imread('employee_original.jpg')
                img_masked = cv2.imread('employee_masked.jpg')
                # print(img.shape)
                faces = face_detector.detectMultiScale(img, 1.3)
                x, y, w, h = faces[0]
                face_orig = img[y:y + h, x:x + w]
                face_masked = img_masked[y:y + h, x:x + w]

                if not os.path.exists('./image_faces/'):
                    os.makedirs('./image_faces/')
                cv2.imwrite(f'./image_faces/face_{self.ui.fname.text()}_{self.ui.lname.text()}.jpg', face_masked)

                # add to database
                # temp_var = self.ui.birthday.text()
                status = self.insert_to_database(self.ui.fname.text(), self.ui.lname.text(), self.ui.code.text(),
                                                 self.ui.birthday.text(),
                                                 f'face_{self.ui.fname.text()}_{self.ui.lname.text()}.jpg')
                if status:
                    self.ui.lbl_error.setText('successfully added to database.')
                else:
                    self.ui.lbl_error.setText('there is a problem with database.')


    def insert_to_database(self, name, family, code, birthday, image_path):
        print(name, family, code, birthday, image_path)
        try:
            my_con = connect('employee.db')
            my_cursor = my_con.cursor()
            my_cursor.execute(f"INSERT INTO employee(fname, lname, code, birthday, image_path) "
                              f"VALUES('{name}','{family}','{code}','{birthday}', '{image_path}')")
            my_con.commit()
            my_con.close()
            return True
        except:
            return False

    def db_fetch(self):
        my_con = connect('employee.db')
        my_cursor = my_con.cursor()
        my_cursor.execute("SELECT * FROM time")
        result = my_cursor.fetchall()
        my_con.close()
        return result

    def show_yellow(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.yellow)

    def back_to_home(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.home)

    def camera(self):
        # self.ui.stackedWidget.setCurrentWidget(self.ui.blue)
        self.webcam.frame_signal.connect(self.show_blue)
        self.webcam.start()

    def show_blue(self, frame, frame_masked):
        self.ui.stackedWidget.setCurrentWidget(self.ui.blue)
        frame = convertCvImage2QtImage(frame)
        self.ui.video_main.setPixmap(frame)

    # def record_picture(self):
    #     self.picture_signal.emit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
