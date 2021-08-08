import sys
import dlib
import numpy as np
import os

import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from gui import Ui_MainWindow


class Main:
    def __init__(self):
        self.image = ""
        self.image_path = ""
        self.camera_streaming = False

        # Main Window initializing
        self.main_window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)

        # Loading the face classifier
        self.face_detector = dlib.get_frontal_face_detector()

        # self.Emotion recognition model
        self.emotion_model = Sequential()

        self.emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Dropout(0.25))

        self.emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Dropout(0.25))

        self.emotion_model.add(Flatten())
        self.emotion_model.add(Dense(1024, activation='relu'))
        self.emotion_model.add(Dropout(0.5))
        self.emotion_model.add(Dense(8, activation='softmax'))
        self.emotion_model.load_weights('emotion_model.h5')

        cv2.ocl.setUseOpenCL(True)

        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Emoji", 3: "Fearful", 4: "Happy", 5: "Neutral", 6: "Sad",
                             7: "Surprised"}
        self.emoji_dist = {0: "angry.png", 1: "disgust.png", 2: "emoji.jpg", 3: "fear.png", 4: "happy.png",
                           5: "natural.png", 6: "sad.png", 7: "surprised.png"}

        self.ui.pushButton_3.clicked.connect(self.upload_image)
        self.ui.pushButton_4.clicked.connect(self.predict)
        self.ui.pushButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))
        self.ui.pushButton_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_2))
        self.ui.pushButton_5.clicked.connect(self.track_over_camera)

    def upload_image(self):
        # open the dialogue box to select the file
        options = QtWidgets.QFileDialog.Options()

        # open the Dialogue box to get the images paths
        image_path = QtWidgets.QFileDialog.getOpenFileName(caption="Select the image", directory="",
                                                           filter="Image Files (*.jpg);;Image Files (*.png);;All "
                                                                  "files (*.*)",
                                                           options=options)

        # If user don't select any image then return without doing any thing
        if image_path[0] == '':
            return

        # Load the image and appear it on the screen
        self.image = cv2.imread(image_path[0])
        resize = cv2.resize(self.image, (280, 280))
        result = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
        height, width, channel = result.shape
        step = channel * width
        qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
        self.ui.label_3.setPixmap(QtGui.QPixmap(qImg))

        self.image_path = image_path[0]

    def predict(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        pred = ""
        print(len(faces))

        # Check face or not found
        if len(faces) != 0:

            # Now get all the face in the frame
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                # Now get the reign of interest of the face and get the prediction over that face
                roi = gray[y1:y2, x1:x2]

                try:
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0)
                    prediction = self.emotion_model.predict(cropped_img)
                    pred = int(np.argmax(prediction))
                    emotion = self.emotion_dict[pred]
                    cv2.putText(self.image, emotion, (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                except Exception as e:
                    print(e)

                # Draw a blue box over the face
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            resize = cv2.resize(self.image, (280, 280))
            result = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
            height, width, channel = result.shape
            step = channel * width
            qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
            self.ui.label_3.setPixmap(QtGui.QPixmap(qImg))

            emotion = self.emotion_dict[pred]
            print(emotion)
            emoji_pic = cv2.imread(os.path.join("emoji", self.emoji_dist[pred]))
            resize = cv2.resize(emoji_pic, (280, 280))
            result = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
            height, width, channel = result.shape
            step = channel * width
            qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
            self.ui.label_6.setPixmap(QtGui.QPixmap(qImg))
        else:
            _translate = QtCore.QCoreApplication.translate
            self.ui.label_4.setStyleSheet("border-radius: 15px;\n"
                                          "color:#4161AD;\n"
                                          "background:none;\n"
                                          "border:2px solid #4161AD;")
            self.ui.label_4.setText(
                _translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; "
                                         "font-weight:600;\">No Face</span></p></body></html>"))


    def track_over_camera(self):
        if self.ui.pushButton_5.text() == "Start Camera":
            self.camera_streaming = True
            self.ui.pushButton_5.setText("Stop Camera")

            video = cv2.VideoCapture("angry.mp4")

            # loop for video stream frames
            while True:
                # read the frame from the camera
                ret, frame = video.read()

                # Proceed if the frame is taken
                if ret:

                    # Convert the frame into gray scale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector(gray)
                    pred = ""

                    # Check face or not found
                    if len(faces) != 0:

                        # Now get all the face in the frame
                        for face in faces:
                            x1 = face.left()
                            y1 = face.top()
                            x2 = face.right()
                            y2 = face.bottom()

                            # Draw a blue box over the face
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # Now get the reign of interest of the face and get the prediction over that face
                            roi = gray[y1:y2, x1:x2]

                            try:
                                # Cropped images
                                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0)
                                prediction = self.emotion_model.predict(cropped_img)

                                pred = int(np.argmax(prediction))
                                emotion = self.emotion_dict[pred]
                                cv2.putText(frame, emotion, (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                            2)
                                print(pred)
                            except:
                                continue
                    else:
                        _translate = QtCore.QCoreApplication.translate
                        self.ui.label_7.setStyleSheet("border-radius: 15px;\n"
                                                      "color:#4161AD;\n"
                                                      "background:rgb(255,255,255);\n"
                                                      "border:2px solid #4161AD;")
                        self.ui.label_7.setText(_translate("MainWindow",
                                                           "<html><head/><body><p><span style=\" font-size:14pt; "
                                                           "font-weight:600;\">Emoji Frame</span></p></body></html>"))
                        cv2.putText(frame, "No Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
                    try:
                        resize = cv2.resize(frame, (280, 280))
                    except:
                        continue
                    result = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
                    height, width, channel = result.shape
                    step = channel * width
                    qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
                    self.ui.label_4.setPixmap(QtGui.QPixmap(qImg))

                    if pred != "":
                        emotion = self.emotion_dict[pred]
                        print(emotion)
                        emoji_pic = cv2.imread(os.path.join("emoji", self.emoji_dist[pred]))
                        try:
                            resize = cv2.resize(emoji_pic, (280, 280))
                        except:
                            continue
                        result = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
                        height, width, channel = result.shape
                        step = channel * width
                        qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
                        self.ui.label_7.setPixmap(QtGui.QPixmap(qImg))

                        # wait for 100 milliseconds
                    key = cv2.waitKey(1)

                    if key == ord('q'):
                        break

                    if not self.camera_streaming:
                        break
            _translate = QtCore.QCoreApplication.translate
            self.ui.label_4.setStyleSheet("border-radius: 15px;\n"
                                          "color:#4161AD;\n"
                                          "background:none;\n"
                                          "border:2px solid #4161AD;")
            self.ui.label_7.setStyleSheet("border-radius: 15px;\n"
                                          "color:#4161AD;\n"
                                          "background:rgb(255,255,255);\n"
                                          "border:2px solid #4161AD;")

            self.ui.label_4.setText(_translate("MainWindow",
                                               "<html><head/><body><p><span style=\" font-size:14pt; "
                                               "font-weight:600;\">Video Frame</span></p></body></html>"))
            self.ui.label_7.setText(_translate("MainWindow",
                                               "<html><head/><body><p><span style=\" font-size:14pt; "
                                               "font-weight:600;\">Emoji Frame</span></p></body></html>"))

            video.release()
            cv2.destroyAllWindows()
        else:
            self.camera_streaming = False
            self.ui.pushButton_5.setText("Start Camera")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.main_window.show()
    sys.exit(app.exec_())
