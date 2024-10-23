import os
import sys
import cv2
import time
from collections import Counter
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import A_identify


class MainUi(QtWidgets.QMainWindow):
    # ================================ Realize the mouse long press to move the window function ================================#
    def mousePressEvent(self, event):
        self.press_x = event.x()  # Record mouse coordinates
        self.press_y = event.y()

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        if 0 < x < 1200 and 0 < y < 60:
            move_x = x - self.press_x
            move_y = y - self.press_y
            position_x = self.frameGeometry().x() + move_x
            position_y = self.frameGeometry().y() + move_y  # Calculates the position of the main window on the desktop after the move
            self.move(position_x, position_y)  # Move main window

    def __init__(self):
        # ================================ parameter definition ================================#
        self.press_x = 0
        self.press_y = 0
        self.identify_api = A_identify.Identify()   # Call identification API
        self .input_image = None                    # input image
        self.output_image = None                    # output image
        self.output_video = None                    # Output video
        self.identify_labels = []                   # result
        self.save_video_flag = False                # Save the video flag bit
        self.save_path = "./A_output/"              # Save the path directory
        self.present_time = QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')  # time
        # ================================ GUI ================================#
        super(MainUi, self).__init__()
        self.resize(1060, 600)  # Interface size
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.ui_title = "Helmet detection system"
        self.timer_time = QTimer()  # Set timer 1
        self.timer_time.timeout.connect(self.update_time)  # Call the time display function periodically
        self.timer_time.start(1000)  # This command is executed every 1000ms/1s
        self.timer_video = QTimer()  # Set timer 2
        self.timer_video.timeout.connect(self.show_video)  # Call the time display function periodically
        # ===================================== Upper navigation bar area============================= #

        self.label = QLabel(self)
        self.label.setText(self.ui_title + "     " + self.present_time)
        self.label.setFixedSize(860, 60)
        self.label.move(0, 0)
        self.label.setStyleSheet("QLabel{padding-left:30px;background:#303030;color:#ffffff;border:none;"
                                 "font-weight:600;font-size:18px;font-family:'微软雅黑'; }")
        # Navigation exit button
        self.b_exit1 = QPushButton(self)
        self.b_exit1.setText("log out")
        self.b_exit1.resize(200, 60)
        self.b_exit1.move(860, 0)
        self.b_exit1.setStyleSheet("QPushButton{background:#303030;text-align:center;border:none;"
                                   "font-weight:600;color:#909090;font-size:15px;}")
        self.b_exit1.setCursor(Qt.PointingHandCursor)
        self.b_exit1.clicked.connect(self.close)
        # ================================= Left parameter control area============================================ #

        self.left_widget = QWidget(self)
        self.left_widget.resize(200, 198)
        self.left_widget.move(0, 60)
        self.left_widget.setStyleSheet("QWidget{background:#ffffff;border:none;}")
        # Confidence threshold text box
        self.conf_label = QLabel(self.left_widget)
        self.conf_label.setText("conf")
        self.conf_label.setFixedSize(190, 30)
        self.conf_label.move(5, 5)
        self.conf_label.setStyleSheet("QLabel{font-size: 18px;color:#999999;font-weight:600;font-family:'微软雅黑'; }")
        # Confidence threshold adjustment box
        self.conf_spin_box = QDoubleSpinBox(self.left_widget)
        self.conf_spin_box.resize(55, 25)
        self.conf_spin_box.move(5, 40)
        self.conf_spin_box.setMinimum(0.0)  # min
        self.conf_spin_box.setMaximum(1.0)  # mix
        self.conf_spin_box.setSingleStep(0.01)
        self.conf_spin_box.setValue(self.identify_api.conf_thres)  # current value
        self.conf_spin_box.setStyleSheet("QDoubleSpinBox{background:#ffffff;color:#999999;font-size:14px;"
                                         "font-weight:600;border: 1px solid #dddddd;}")
        self.conf_spin_box.valueChanged.connect(self.change_conf_spin_box)
        # Confidence threshold scroll bar
        self.conf_slider = QSlider(Qt.Horizontal, self.left_widget)
        self.conf_slider.resize(130, 25)
        self.conf_slider.move(65, 40)
        self.conf_slider.setMinimum(0)  # min
        self.conf_slider.setMaximum(100)  # max
        self.conf_slider.setSingleStep(1)
        self.conf_slider.setValue(int(self.identify_api.conf_thres * 100))  # current value
        self.conf_slider.setStyleSheet("QSlider::groove:horizontal{border:1px solid #999999;height:25px;}"
                                       "QSlider::handle:horizontal{background:#ffcc00;width:24px;border-radius:12px;}"
                                       "QSlider::add-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                       "x2:0,y2:0,stop:0 #d9d9d9,stop:0.25 #d9d9d9,stop:0.5 #d9d9d9,stop:1 #d9d9d9);}"
                                       "QSlider::sub-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                       "x2:0,y2:0,stop:0 #777777,stop:0.25 #777777,stop:0.5 #777777,stop:1 #777777);}")
        self.conf_slider.valueChanged.connect(self.change_conf_slider)
        #Cross and compare IoU threshold text box
        self.iou_label = QLabel(self.left_widget)
        self.iou_label.setText("IoU") #iou
        self.iou_label.setFixedSize(190, 30)
        self.iou_label.move(5, 70)
        self.iou_label.setStyleSheet("QLabel{font-size: 18px;color:#999999;font-weight:600;font-family:'微软雅黑'; }")
        # Crossover ratio IoU threshold adjustment frame
        self.iou_spin_box = QDoubleSpinBox(self.left_widget)
        self.iou_spin_box.resize(55, 25)
        self.iou_spin_box.move(5, 105)
        self.iou_spin_box.setMinimum(0.0)  # min
        self.iou_spin_box.setMaximum(1.0)  # mix
        self.iou_spin_box.setSingleStep(0.01)
        self.iou_spin_box.setValue(self.identify_api.iou_thres)
        self.iou_spin_box.setStyleSheet("QDoubleSpinBox{background:#ffffff;color:#999999;font-size:14px;"
                                        "font-weight:600;border: 1px solid #dddddd;}")
        self.iou_spin_box.valueChanged.connect(self.change_iou_spin_box)
        # Cross the IoU threshold scroll bar
        self.iou_slider = QSlider(Qt.Horizontal, self.left_widget)
        self.iou_slider.resize(130, 25)
        self.iou_slider.move(65, 105)
        self.iou_slider.setMinimum(0)  # min
        self.iou_slider.setMaximum(100)  # mix
        self.iou_slider.setSingleStep(1)
        self.iou_slider.setValue(int(self.identify_api.iou_thres * 100))
        self.iou_slider.setStyleSheet("QSlider::groove:horizontal{border:1px solid #999999;height:25px;}"
                                      "QSlider::handle:horizontal{background:#ffcc00;width:24px;border-radius:12px;}"
                                      "QSlider::add-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                      "x2:0,y2:0,stop:0 #d9d9d9,stop:0.25 #d9d9d9,stop:0.5 #d9d9d9,stop:1 #d9d9d9);}"
                                      "QSlider::sub-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                      "x2:0,y2:0,stop:0 #777777,stop:0.25 #777777,stop:0.5 #777777,stop:1 #777777);}")
        self.iou_slider.valueChanged.connect(self.change_iou_slider)
        # Save the detection result text box
        self.save_label = QLabel(self.left_widget)
        self.save_label.setText("Test result saving")
        self.save_label.resize(190, 25)
        self.save_label.move(5, 135)
        self.save_label.setStyleSheet("QLabel{font-size: 18px;color:#999999;font-weight:600;font-family:'微软雅黑'; }")
        # Save the check result option box
        self.save_button_yes = QRadioButton(self.left_widget)
        self.save_button_yes.setText("  Yes")
        self.save_button_yes.resize(70, 25)
        self.save_button_yes.move(5, 165)
        self.save_button_yes.setStyleSheet(
            "QRadioButton{font-size: 16px;color:#999999;font-weight:600;font-family:'黑体'; }")
        self.save_button_no = QRadioButton(self.left_widget)
        self.save_button_no.setText("  No")
        self.save_button_no.resize(70, 25)
        self.save_button_no.move(80, 165)
        self.save_button_no.setStyleSheet(
            "QRadioButton{font-size: 16px;color:#999999;font-weight:600;font-family:'黑体'; }")
        self.save_button_no.setChecked(False)  # Save selected by default
        # =================================================== left  result============================================= #
        self.result_label = QLabel(self)
        self.result_label.setText("testing result")
        self.result_label.setFixedSize(200, 32)
        self.result_label.move(0, 262)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("QLabel{background:#ffffff;color:#999999;border:none;font-weight:600;"
                                        "font-size:18px;font-family:'黑体';}")
        self.result = QLabel(self)
        self.result.setText("")
        self.result.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
        self.result.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result.setStyleSheet("QLabel{background:#ffffff;color:#999999;border:none;font-weight:600;"
                                  "font-size:15px;font-family:'黑体';padding:5px;}")
        self.scroll_area = QScrollArea(self)
        self.scroll_area.resize(200, 305)
        self.scroll_area.move(0, 295)
        self.scroll_area.setWidget(self.result)
        self.scroll_area.setWidgetResizable(True)  # 设置 QScrollArea 大小可调整
        self.scroll_area.setStyleSheet("QScrollArea{border:none;}")
        # ========================================================= Right side display area========================================== #
        # test speed
        self.identify_v = QLabel(self)
        self.identify_v.setText("test the speed：")
        self.identify_v.setFixedSize(820, 20)
        self.identify_v.move(220, 65)
        self.identify_v.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.identify_v.setStyleSheet("QLabel{color:#999999;font-weight:600;font-size:15px;font-family:'黑体';}")
        # Original picture display area
        self.input_img = QLabel(self)
        self.input_img.setText("input display area")
        self.input_img.setFixedSize(400, 400)
        self.input_img.move(220, 85)
        self.input_img.setAlignment(Qt.AlignCenter)
        self.input_img.setStyleSheet("QLabel{border: 2px solid gray;font-size:30px;font-family:'黑体';color:#999999;}")
        # test picture
        self.output_img = QLabel(self)
        self.output_img.setText("output display area")
        self.output_img.setFixedSize(400, 400)
        self.output_img.move(640, 85)
        self.output_img.setAlignment(Qt.AlignCenter)
        self.output_img.setStyleSheet("QLabel{border: 2px solid gray;font-size:30px;font-family:'黑体';color:#999999;}")
        # ================================================== Right button area =================================== #
        # Right button 1(Image detection)
        self.function1 = QPushButton(self)
        self.function1.setText("Image detection")
        self.function1.resize(250, 60)
        self.function1.move(220, 510)
        self.function1.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                     "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                     "QPushButton:hover{background:#e6e6e6;}")
        self.function1.setFocusPolicy(Qt.NoFocus)
        self.function1.setCursor(Qt.PointingHandCursor)
        self.function1.clicked.connect(self.show_image)
        # Right button 2(Video detection)
        self.function2 = QPushButton(self)
        self.function2.setText("video detection")
        self.function2.resize(250, 60)
        self.function2.move(505, 510)
        self.function2.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                     "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                     "QPushButton:hover{background:#e6e6e6;}")
        self.function2.setFocusPolicy(Qt.NoFocus)
        self.function2.setCursor(Qt.PointingHandCursor)
        self.function2.clicked.connect(self.video_identify)
        # Right button 3(camera detection)
        self.function3 = QPushButton(self)
        self.function3.setText("camera detection")
        self.function3.resize(250, 60)
        self.function3.move(790, 510)
        self.function3.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                     "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                     "QPushButton:hover{background:#e6e6e6;}")
        self.function3.setFocusPolicy(Qt.NoFocus)
        self.function3.setCursor(Qt.PointingHandCursor)
        self.function3.clicked.connect(self.camera_identify)

    # ================================ Function area ================================#
    #  Show current time
    def update_time(self):
        self.present_time = QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')
        self.label.setText(self.ui_title + "     " + self.present_time)

    #   Adjust the frame to change the detection confidence
    def change_conf_spin_box(self):
        conf_thres = round(self.conf_spin_box.value(), 2)
        self.conf_slider.setValue(int(conf_thres * 100))
        self.identify_api.conf_thres = conf_thres

    #  Scrollbar changes the detection confidence
    def change_conf_slider(self):
        conf_thres = round(self.conf_slider.value() * 0.01, 2)
        self.conf_spin_box.setValue(conf_thres)
        self.identify_api.conf_thres = conf_thres

    #  Adjust the frame to change the detection intersection ratio
    def change_iou_spin_box(self):
        iou_thres = round(self.iou_spin_box.value(), 2)
        self.iou_slider.setValue(int(iou_thres * 100))
        self.identify_api.iou_thres = iou_thres

    #  The scroll bar changes the detection intersection ratio
    def change_iou_slider(self):
        iou_thres = round(self.iou_slider.value() * 0.01, 2)
        self.iou_spin_box.setValue(iou_thres)
        self.identify_api.iou_thres = iou_thres

    #   Image detection
    def show_image(self):
        image_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open picture", "./", "*.jpg;*.png;;All Files(*)")
        #    The selected image name must be at least 5
        if len(image_path) >= 5:
            self.input_image = cv2.imread(image_path)
            start_time = time.time()
            self.input_image, self.output_image, self.identify_labels = self.identify_api.show_frame(self.input_image, False)
            if self.output_image is not None:
                # ======================================== Displays the interface and results ================================================================== #
                show_input_img = self.change_image(self.input_image)
                show_output_img = self.change_image(self.output_image)
                # The detection image screen is displayed on the interface
                show_input_img = cv2.cvtColor(show_input_img, cv2.COLOR_BGR2RGB)
                show_input_img = QImage(show_input_img.data, show_input_img.shape[1], show_input_img.shape[0],
                                        show_input_img.shape[1] * 3, QImage.Format_RGB888)
                self.input_img.setPixmap(QPixmap.fromImage(show_input_img))
                show_output_img = cv2.cvtColor(show_output_img, cv2.COLOR_BGR2RGB)
                show_output_img = QImage(show_output_img.data, show_output_img.shape[1], show_output_img.shape[0],
                                         show_output_img.shape[1] * 3, QImage.Format_RGB888)
                self.output_img.setPixmap(QPixmap.fromImage(show_output_img))
                # Display test speed and test results in the results display area
                end_time = time.time()  # End of record time
                execution_time = str(round(end_time - start_time, 2)) + " s"
                self.identify_v.setText("test the speed： " + execution_time)
                identify_result = ""
                counter = Counter(self.identify_labels)  # Counter Counts the number of occurrences of elements
                for element, count in counter.items():
                    identify_result = identify_result + str(element) + ": " + str(count) + "\n"
                self.result.setText(identify_result)
                # =================================================== Save the result =========================================== #
                if self.save_button_yes.isChecked():
                    file_path = os.path.join(self.save_path, "images/" +
                                             QDateTime.currentDateTime().toString('yyyy_MM_dd_hh_mm_ss') + ".jpg")
                    cv2.imwrite(file_path, self.output_image)
                    msg_box = QMessageBox()
                    msg_box.setText("The detection result is saved in the./A output path!")
                    msg_box.exec_()
            else:
                self.reset()
        else:
            self.reset()

    # video detection
    def video_identify(self):
        if self.function2.text() == "Enabling Video detection" and not self.timer_video.isActive():
            video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "open Video", "", "*.mp4;*.avi;;All Files(*)")
            if len(video_path) > 5:
                flag = self.identify_api.cap.open(video_path)
                if flag is False:
                    QtWidgets.QMessageBox.warning(
                        self, u"Warning", u"Failed to open video", buttons=QtWidgets.QMessageBox.Ok,
                        defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_video.start(30)
                    self.function1.setDisabled(True)
                    self.function3.setDisabled(True)
                    self.function1.setStyleSheet("QPushButton{background:#e6e6e6;color:#999999;font-weight:600;"
                                                 "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                                 "QPushButton:hover{background:#e6e6e6;}")
                    self.function3.setStyleSheet("QPushButton{background:#e6e6e6;color:#999999;font-weight:600;"
                                                 "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                                 "QPushButton:hover{background:#e6e6e6;}")
                    self.function2.setText("Disable video detection")
            else:
                self.reset()
        else:
            self.identify_api.cap.release()
            self.timer_video.stop()
            self.function1.setDisabled(False)
            self.function3.setDisabled(False)
            self.function1.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                         "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                         "QPushButton:hover{background:#e6e6e6;}")
            self.function3.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                         "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                         "QPushButton:hover{background:#e6e6e6;}")
            self.function2.setText("Enabling Video detection")
            self.reset()

    # Camera detection
    def camera_identify(self):
        if self.function3.text() == "Enable camera detection" and not self.timer_video.isActive():
            flag = self.identify_api.cap.open(0)# The first local camera is used by default
            if flag is False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"Failed to open the camera", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_video.start(30)
                self.function1.setDisabled(True)
                self.function2.setDisabled(True)
                self.function1.setStyleSheet("QPushButton{background:#e6e6e6;color:#999999;font-weight:600;"
                                             "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                             "QPushButton:hover{background:#e6e6e6;}")
                self.function2.setStyleSheet("QPushButton{background:#e6e6e6;color:#999999;font-weight:600;"
                                             "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                             "QPushButton:hover{background:#e6e6e6;}")
                self.function3.setText("Turn off camera detection")
        else:
            self.identify_api.cap.release()
            self.timer_video.stop()
            self.function1.setDisabled(False)
            self.function2.setDisabled(False)
            self.function1.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                         "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                         "QPushButton:hover{background:#e6e6e6;}")
            self.function2.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                         "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                         "QPushButton:hover{background:#e6e6e6;}")
            self.function3.setText("Enable camera detection")
            self.reset()

    # Display images and display results (video and camera)
    def show_video(self):
        start_time = time.time()  # Start of record time
        self.input_image, self.output_image, self.identify_labels = self.identify_api.show_frame(None, True)
        if self.output_image is not None:
            # ===================================================== Save the result ================================================== #
            if self.save_button_yes.isChecked():
                # Save video
                if self.save_video_flag is False:
                    self.save_video_flag = True
                    fps = self.identify_api.cap.get(cv2.CAP_PROP_FPS)
                    w = int(self.identify_api.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self.identify_api.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if self.function2.text() == "Disable video detection":
                        save_path = self.save_path + "/videos/saved_" + QDateTime.currentDateTime().toString(
                            'yyyy_MM_dd_hh_mm_ss') + ".mp4"
                    else:
                        save_path = self.save_path + "/camera/saved_" + QDateTime.currentDateTime().toString(
                            'yyyy_MM_dd_hh_mm_ss') + ".mp4"
                    self.output_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                self.output_video.write(self.output_image)
            # ========================== Displays the interface and results ===================== #
            show_input_img = self.change_image(self.input_image)
            show_output_img = self.change_image(self.output_image)
                # The detection image screen is displayed on the interface
            show_input_img = cv2.cvtColor(show_input_img, cv2.COLOR_BGR2RGB)
            show_input_img = QImage(show_input_img.data, show_input_img.shape[1], show_input_img.shape[0],
                                    show_input_img.shape[1] * 3, QImage.Format_RGB888)
            self.input_img.setPixmap(QPixmap.fromImage(show_input_img))
            show_output_img = cv2.cvtColor(show_output_img, cv2.COLOR_BGR2RGB)
            show_output_img = QImage(show_output_img.data, show_output_img.shape[1], show_output_img.shape[0],
                                     show_output_img.shape[1] * 3, QImage.Format_RGB888)
            self.output_img.setPixmap(QPixmap.fromImage(show_output_img))
            # Display test speed and test results in the results display area
            end_time = time.time()  # End of record time
            execution_time = str(round(end_time - start_time, 2)) + " s"
            self.identify_v.setText("test speed： " + execution_time)
            identify_result = ""
            counter = Counter(self.identify_labels)
            for element, count in counter.items():
                identify_result = identify_result + str(element) + ": " + str(count) + "\n"
            self.result.setText(identify_result)
        else:
            self.timer_video.stop()
            self.function1.setDisabled(False)
            self.function2.setDisabled(False)
            self.function3.setDisabled(False)
            self.function1.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                         "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                         "QPushButton:hover{background:#e6e6e6;}")
            self.function2.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                         "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                         "QPushButton:hover{background:#e6e6e6;}")
            self.function3.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                         "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                         "QPushButton:hover{background:#e6e6e6;}")
            self.function2.setText("Enabling Video detection")
            self.function3.setText("Enabling Camera detection")
            self.reset()

    # Change the image size displayed in the interface
    @staticmethod
    def change_image(input_image):
        if input_image is not None:
            # Replace the adaptive size display on the interface
            wh = float(int(input_image.shape[0]) / int(input_image.shape[1]))
            show_wh = 1
            if int(input_image.shape[0]) > 400 or int(input_image.shape[1]) > 400:
                if show_wh - wh < 0:
                    h = 400
                    w = int(h / wh)
                    output_image = cv2.resize(input_image, (w, h))
                else:
                    w = 400
                    h = int(w * wh)
                    output_image = cv2.resize(input_image, (w, h))
            else:
                output_image = input_image
            return output_image
        else:
            return input_image

    # Clear reset data
    def reset(self):
        if self.save_button_yes.isChecked() and self.output_video is not None:
            self.output_video.release()
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setText("The detection result is saved in the./A output path!")
            # 显示消息框
            msg_box.exec_()
        self.input_image = None  # input image
        self.output_image = None  # output image
        self.output_video = None  # Output video
        self.identify_labels = []  # testing result
        self.save_video_flag = False  # Save video flag
        self.input_img.clear()  # Clear the input image display area
        self.input_img.setText("Input display area")
        self.output_img.clear()  # Clear the output image display area
        self.output_img.setText("output display area")
        self.identify_v.setText("test the speed：")
        self.result.clear()  # Clear the results area
        self.save_button_no.setChecked(True)  # Check not save


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainUi = MainUi()
    mainUi.show()
    sys.exit(app.exec_())
