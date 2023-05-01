# import system module
import sys, serial, serial.tools.list_ports, warnings

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from Design2 import *

import imutils
import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import serial

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

frame_num = 0
config = None
input_size = None
infer = None
session = None
STRIDES = None
ANCHORS = None
NUM_CLASS = None
XYSCALE = None
counter_btn = 0

# arduino_ports = [
    # p.device
    # for p in serial.tools.list_ports.comports()
    # if 'Arduino' in p.description  # may need tweaking to match new arduinos
# ]
# if not arduino_ports:
    # raise IOError("No Arduino found")
# if len(arduino_ports) > 1:
    # warnings.warn('Multiple Arduinos found - using the first')
    
class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.start_bt.clicked.connect(self.controlTimer)
        self.ui.connect_bt.clicked.connect(self.connection)

    def connection(self):
        global counter_btn
        counter_btn = counter_btn+1
        if counter_btn == 1:
            try:
                COM = self.ui.COM_Box.currentText()
                BAUDRATE = self.ui.Baudrate_Box.currentText()
                utils.bukakoneksi(COM,BAUDRATE)
                self.ui.connect_bt.setText("Disconnect Serial")
                self.ui.plainTextEdit_print.appendPlainText(str("Serial Connected"))
            except:
                counter_btn = 0
                pass
        elif counter_btn == 2:
            try:
                counter_btn = 0
                utils.tutupkoneksi()
                self.ui.connect_bt.setText("Connect to Serial")
                self.ui.plainTextEdit_print.appendPlainText(str("Serial Disconnected"))
            except:
                counter_btn = 0
                pass

    # view camera
    def viewCam(self):
        # read image in BGR format
        global infer
        global frame_num
        global input_size
        global STRIDES
        global ANCHORS
        global NUM_CLASS
        global XYSCALE
        
        global scale
        global prev_gray
        global mask
        
        ret, frame = self.cap.read()
        if ret:
            #frame = imutils.resize(frame, width=640)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            return
            
        # Convert new frame format`s to gray scale and resize gray frame obtained
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=scale, fy=scale)

        # Calculate dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 0)
        # Compute the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Set image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Set image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        
        # Resize frame size to match dimensions
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Open a new window and displays the output frame
        dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)
        #cv2.imshow("Dense optical flow", dense_flow)
        # Update previous frame
        prev_gray = gray    
            
            
        frame = imutils.resize(frame, width=861)
        #frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for _, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        motion = np.sum(magnitude)
        if (motion> 200000):
            print("motion detected")
            allowed_classes = list(class_names.values())
        else:
            allowed_classes = []
        image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)   
      
        fps = 1.0 / (time.time() - start_time)
        result = np.asarray(image)
        # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Result", result)
        
        # get image infos
        height, width, channel = result.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, "FPS: %.2f" % fps, (0, 70), font, 1, (255, 255, 255), 3, cv2.LINE_AA)
        self.ui.image_label_cam.setPixmap(QPixmap.fromImage(qImg))
        self.ui.plainTextEdit_print.appendPlainText(str(utils.warning))
        

    # start/stop timer
    def controlTimer(self):
        global scale
        global prev_gray
        global mask
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            
            # Read first frame
            ret, first_frame = self.cap.read()
            # Scale and resize image
            resize_dim = 600
            max_dim = max(first_frame.shape)
            scale = resize_dim/max_dim
            first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
            # Convert to gray scale 
            prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            # Create mask
            mask = np.zeros_like(first_frame)
            # Sets image saturation to maximum
            mask[..., 1] = 255
            
            # update control_bt text
            self.ui.start_bt.setText("Stop")
            self.ui.connect_bt.setEnabled(False)
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.start_bt.setText("Start")
            self.ui.connect_bt.setEnabled(True)
            
# def bukakoneksi(COM,BAUDRATE,STATE):
    # if STATE == 1:
        # ser = serial.Serial(COM, BAUDRATE, timeout=1)
        # print("Serial Connected")
    # elif STATE == 2:
        # ser.close()
        # print("Serial Disconnected")
    # pass    
             

def main(_argv):
    global config
    global input_size
    global infer
    global session
    global STRIDES
    global ANCHORS
    global NUM_CLASS
    global XYSCALE

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    print("Initialized")

    

    app = QApplication(_argv)
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass