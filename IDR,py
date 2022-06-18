"""
This file was written by Eng. Abduallah Damash, Team Leader of Fearless Five Team Group
Except the part of the face recognition, was writen originally by Eng. baran baris yalcin.
This file represents all the subsystem of identity recognition of the final product.
It firstly implements the face recognition function to identify the people around the house.
Secondly, it detects the emotion of the person, and report it is angr or fear.
Finally, it detects the tools and the wild animals.
"""
import re
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
from imutils import paths
import numpy as np
import time
import face_recognition
import os
import cv2
from deepface import DeepFace
# noinspection PyPackageRequirements
import torch
import warnings
# noinspection PyPackageRequirements
import tensorflow as tf
from sphinx.util import requests
from pyzbar.pyzbar import decode
import pyshine as ps  
from threading import Thread
import RPi.GPIO as GPIO
# noinspection SpellCheckingInspection
HTML = """
<html>
<head>
<title>PyShine Live Streaming</title>
</head>

<body>
<center><h1> Fearless Five  </h1></center>
<center><img src="stream.mjpg" width='640' height='480' autoplay playsinline></center>
</body>
</html>
"""
capture = cv2.VideoCapture(0)
yolov5_source_path = r"/home/abood/last/ultralytics_yolov5"
tool_model_path = r"/home/abood/last/best-yolo-weights-V.pt"
######################################################################################################
# Firebase Database Parameters
database_url = 'https://fearless-five-default-rtdb.firebaseio.com/'
cert_path = r"/home/abood/last/fearless-five-firebase-adminsdk.json"

try:
    cred = credentials.Certificate(cert_path)
    # Initialize the app with a service account, granting admin privileges
    firebase_admin.initialize_app(cred, {
        'databaseURL': database_url
    })
    ref = db.reference('/IDR')
    new_add = db.reference('IDR/New')
except RuntimeError as e:
    print(e)

######################################################################################################
# tool_model = torch.hub.load(yolov5_source_path, 'custom', path=tool_model_path,
#                             force_reload=True, source='local')
# tool_classes = tool_model.names
# print(tool_classes)  #
# if False:
#     model = torch.hub.load(yolov5_source_path, 'yolov5s', pretrained=True, source='local')
#     yolov5_classes = self.yolov5_model.names
#     print(yolov5_classes)
######################################################################################################
# General Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)

######################################################################################################
# Tensorflow and Warning Parameters
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# noinspection PyTypeChecker,PyUnboundLocalVariable
def stream():
    StreamProps = ps.StreamProps
    StreamProps.set_Page(StreamProps, HTML)
    address = ('192.168.137.193', 9000)  # Enter your IP address
    try:
        StreamProps.set_Mode(StreamProps, 'cv2')
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        capture.set(cv2.CAP_PROP_FPS, 30)
        StreamProps.set_Capture(StreamProps, capture)
        StreamProps.set_Quality(StreamProps, 90)
        server = ps.Streamer(address, StreamProps)
        print('Server started at', 'http://'+address[0]+':'+str(address[1]))
        server.serve_forever()

    except KeyboardInterrupt:
        capture.release()
        server.socket.close()


def send_signals(signal_name, signal_score, signal_type, current_time):
    """
    Takes a signal name and its results as input, and send them to real-time firebase.
    :param signal_name: the name of the signal (emotion, tool, animals).
    :param signal_score: the probability of that signal occurred .
    :param signal_type: the type of the signal.
    :param current_time: the time that the event occurred.
    :return: .
    """
    ref.child(signal_type).set(
        {
            'Time': str(current_time),
            'Signal Name': str(signal_name),
            'Signal Score': str(signal_score)
        }
    )


def send_flag():
    """
    Send the signal case of adding visitor or blacklist to real-time firebase.
    :return: .
    """
    new_add.update(
        {
            'ADD_FLAG': 'false'
        }
    )


def servo(start, end, delay, loop):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(11, GPIO.OUT)

    servo1 = GPIO.PWM(11, 50)
    servo1.start(0)
    for i in range(0, int(loop)):
        for dc in range(int(start), int(end), 1):
            servo1.ChangeDutyCycle(2+(dc/18))
            time.sleep(float(delay))
            servo1.ChangeDutyCycle(0)
            time.sleep(0.3)
            print(dc)

    servo1.stop()
    GPIO.cleanup()


class Faces:
    def __init__(self, blacklist_names, face_dataset_path,
                 visitor_names, owner_names, pets_names, cap):
        # emotion_model_path
        # emotion_model_keras_path
        """
        Initializes the class with required file.
        :param blacklist_names: your blacklist dataset names for face recognitions.
        :param face_dataset_path: your faces' dataset path for face recognitions.
        :param visitor_names: your visitors' dataset names for face recognitions.
        :param owner_names: your owners' dataset names for face recognitions.
        :param pets_names: your pets' dataset names for face recognitions.
        :param cap: camera port.
        """
        ######################################################################################################
        # Paths
        self.face_dataset_path = face_dataset_path
        ######################################################################################################
        # Names and Types
        self.blacklist_names = blacklist_names
        self.visitor_names = visitor_names
        self.owner_names = owner_names
        self.pets_names = pets_names
        self.criminal_names = ['EnderPKK', 'AhmetPKK']
        self.person_name = ''
        self.img_url = ''
        self.add_flag = ''
        self.person_type = ''
        ######################################################################################################
        # Face recognition Parameters
        self.known_encodings = []
        self.known_names = []
        self.faces_data = self.encoding_faces(image_path=self.face_dataset_path)
        self.face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        ######################################################################################################
        self.cap = cap
        self._running = True

    def encoding_faces(self, image_path):
        print("All faces in database are being encoded...")
        image_paths = list(paths.list_images(image_path))

        for (i, imagePath) in enumerate(image_paths):
            name = imagePath.split(os.path.sep)[-1]
            name = os.path.splitext(name)[0]
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model='hog')
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:
                self.known_encodings.append(encoding)
                self.known_names.append(name)
                # if name in self.blacklist_names:
                #     self.blacklist_names.append(name)
                # elif name in self.visitor_names:
                #     self.visitor_names.append(name)
                # elif name in self.owner_names:
                #     self.owner_names.append(name)

        # save encodings along with their names in dictionary data
        encoded_data = {"encodings": self.known_encodings, "names": self.known_names}
        return encoded_data

    # noinspection PyUnusedLocal
    def finding_faces(self, frame, encoded_faces, detected_faces_encoding, detected_faces,
                      blacklist_names, visitor_names, owner_names):
        names = []
        for encoding in detected_faces_encoding:
            matches = face_recognition.compare_faces(encoded_faces["encodings"], encoding)
            name = "Unknown"
            if True in matches:
                matched_ids = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matched_ids:
                    name = encoded_faces["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)
            for ((x, y, w, h), name) in zip(detected_faces, names):
                # rescale the face coordinates
                # draw the predicted face name on the image
                print(name)
                # try:
                #     # noinspection PyTypeChecker
                #     analyze = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                #     cv2.putText(frame, analyze['dominant_emotion'],
                #                 (analyze['region']['x'], analyze['region']['y']),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                #     print(analyze['dominant_emotion'])
                #     best_score = analyze['emotion'][max(analyze['emotion'], key=analyze['emotion'].get)]
                #     if (analyze['dominant_emotion'] in 'angry') or \
                #             (analyze['dominant_emotion'] in 'fear' and name in owner_names):
                #         self.send_signals(analyze['dominant_emotion'], best_score, 'Emotion',
                #                           datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                # except BaseException as err:
                #     print(f"Unexpected {err=}, {type(err)=}")
                if name in blacklist_names:
                    send_signals(name + '  near house', 100, 'Blacklist',
                                 datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                    # cv2.putText(frame, name + ' from Blacklist and near house', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #             (0, 0, 255), 4)
                elif name in self.criminal_names:
                    send_signals(name + ' is Criminal and near house', 100, 'Blacklist',
                                 datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                    # cv2.putText(frame, name + ' is Criminal and near house', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #             (0, 0, 255), 4)
                elif name in visitor_names:
                    send_signals(name + '  near house', 100, 'Visitor',
                                 datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (45, 255, 255), 5)
                    # cv2.putText(frame, name + ' from Visitor and near house', (x, y),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 255, 255), 4)
                elif name in owner_names:  # name != 'Unknown' and
                    send_signals(name + '  near house', 100, 'Owner',
                                 datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                    # cv2.putText(frame, name + ' from Owner and near house', (x, y),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 4)
                else:
                    send_signals(name + '  person near house', 100, 'Unknown',
                                 datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                    # cv2.putText(frame, name + '  person near house', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.5, (0, 255, 0), 4)

        return frame

    def finding_qrcodes(self, frame):
        for barcode in decode(frame):
            data = barcode.data.decode('utf-8')
            print(data)
            data = re.split('0', data)
            data = data[0]
            if data in self.pets_names:
                # pts = np.array([barcode.polygon], np.int32)
                # pts = pts.reshape((-1, 1, 2))
                # cv2.polylines(frame, [pts], True, (255, 0, 0), 5)
                # pts2 = barcode.rect
                # cv2.putText(frame, data + ' from Pets List', (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.9, (255, 0, 0), 2)
                send_signals(data, 100, 'Pets', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        return frame

    def get_frame(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        while self._running:
            start_time = time.time()
            print("System Information ..................")
            print("Owners' Users List ..................")
            print(self.owner_names)
            print("........................... .........")
            print("Visitors' Users List ................")
            print(self.visitor_names)
            print(".................. .........")
            print("Blacklist's Users List ..............")
            print(self.blacklist_names)
            print("........................... .........")
            print("Pets' Users List ....................")
            print(self.pets_names)
            print("........................... .........")
            time1 = time.time()
            print("Reading from Database......")
            flag = new_add.order_by_key().get()
            for key, val in flag.items():
                if key == 'ADD_FLAG':
                    self.add_flag = val
                elif key == 'Link':
                    self.img_url = val
                elif key == 'Type':
                    self.person_type = val
                elif key == 'Name':
                    self.person_name = val
            print(self.person_name, self.img_url, self.add_flag, self.person_type)
            if self.add_flag == 'True':
                time4 = time.time()
                print("Adding User Start now.....")
                response = requests.get(self.img_url)
                database_path = self.face_dataset_path
                name_photo = database_path + "/" + self.person_name + ".jpg"
                file = open(name_photo, "wb")
                file.write(response.content)
                file.close()
                if self.person_type == 'blacklist':
                    self.blacklist_names.append(self.person_name)
                elif self.person_type == 'visitor':
                    self.visitor_names.append(self.person_name)
                elif self.person_type == 'owner':
                    self.owner_names.append(self.person_name)
                self.faces_data = self.encoding_faces(image_path=self.face_dataset_path)
                print(f'Adding User Ending... Total Time:{time.time() - time4} s')
                send_flag()

            elif self.add_flag == 'Pet':
                if self.person_type == 'pet':
                    self.pets_names.append(self.person_name)
                send_flag()
            elif self.add_flag == 'Remove':
                if self.person_type == 'blacklist':
                    self.blacklist_names.remove(self.person_name)
                elif self.person_type == 'visitor':
                    self.visitor_names.remove(self.person_name)
                elif self.person_type == 'owner':
                    self.owner_names.remove(self.person_name)
                elif self.person_type == 'pet':
                    self.pets_names.remove(self.person_name)
                send_flag()
            elif self.add_flag == 'left':
                print("Left")
                servo(1, 2, 0.1, 1)
                # os.system("python servo.py 1 2 0.1 1")
                send_flag()
            elif self.add_flag == 'right':
                print("Right")
                servo(179, 180, 0.1, 1)
                # os.system("python servo.py 179 180 0.1 1")
                send_flag()
            elif self.add_flag == 'center':
                print("Center")
                servo(89, 90, 0.3, 1)
                # os.system("python servo.py 89 90 0.3 1")
                send_flag()

            print(f'Finish Reading from Database....... Total Time:{time.time() - time1} s')

            ret, orgi_frame = self.cap.read()
            if not ret:
                ######################################################################################################
                # First, recognize the faces: (Face Recognitions)
                time2 = time.time()
                print("First Stage starting... Face Recognitions")
                frame_gray = cv2.cvtColor(orgi_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_haar_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5,
                                                                minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

                frame_rgb = cv2.cvtColor(orgi_frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(frame_rgb)
                frame = self.finding_faces(frame=orgi_frame, encoded_faces=self.faces_data,
                                           detected_faces_encoding=encodings, detected_faces=faces,
                                           blacklist_names=self.blacklist_names,
                                           visitor_names=self.visitor_names,
                                           owner_names=self.owner_names)
                print(f'1- Face Recognitions Ending... Total Time:{time.time() - time2} s')

                time3 = time.time()
                print("Second Stage starting... Pets Detection")
                orgi_frame = self.finding_qrcodes(frame)
                print(f'2- Pets Detection Ending... Total Time:{time.time() - time3} s')
                time5 = time.time()
                print("Third Stage starting... Emotion Detection")
                try:
                    # noinspection PyTypeChecker
                    analyze = DeepFace.analyze(orgi_frame, actions=['emotion'], enforce_detection=True)
                    # cv2.putText(frame_resized, analyze['dominant_emotion'],
                    #             (analyze['region']['x'], analyze['region']['y']),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                    print("found emotion is " + analyze['dominant_emotion'])
                    best_score = analyze['emotion'][max(analyze['emotion'], key=analyze['emotion'].get)]
                    # noinspection PyRedundantParentheses
                    if analyze['dominant_emotion'] in ('angry'):  # , 'fear'
                        send_signals('someone is ' + analyze['dominant_emotion'], best_score, 'Emotion',
                                     datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                except BaseException as err:
                    print(f"Unexpected {err=}, {type(err)=}")
                print(f'3- Emotion Detection Ending... Total Time:{time.time() - time5} s')

                end_time = time.time()
                fps = 1 / np.round(end_time - start_time, 2)
                # print(f"Frames Per Second : {fps}")
                # try:
                #     fps1 = int(fps)
                # except BaseException as err:
                #     print(f"Unexpected {err=}, {type(err)=}")
                #     fps1 = fps
                print(f'Total Frame Rates: {fps}')
                # cv2.putText(frame_resized, f'FPS: {fps1}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

#
# class Tool:
#     def __init__(self, cap, confidence, img_size):
#         """
#         Initializes the class with required file.
#         :param confidence: confidence threshold score for running yoloV5 model.
#         :param img_size: The image size that works with custom Yolo v5 model.
#         :param cap: camera port.
#         """
#         self.threshold = confidence
#         self.img_size = img_size
#         self.cap = cap
#         self._running = True
#
#     def plot_boxes(self, model_used, frame, classes):
#         """
#         Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
#         :param model_used: required model to find the score.
#         :param frame: Frame which has been scored.
#         :param classes: model classes to plot the names.
#         :return: Frame with bounding boxes and labels plotted on it.
#         """
#         model_used.to(device)
#         frame = [frame]
#         results = model_used(frame)
#         labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
#         # labels, cord = results
#         n = len(labels)
#         # x_shape, y_shape = frame.shape[1], frame.shape[0]
#         for i in range(n):
#             row = cord[i]
#             if row[4] >= self.threshold:
#                 # x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), \
#                 #                  int(row[2] * x_shape), int(row[3] * y_shape)
#                 # bgr = (0, 255, 0)
#                 # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
#                 # cv2.putText(frame, classes[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
#                 try:
#                     if classes[int(labels[i])] in ('ak47', 'baseball-bat', 'knife', 'pistol', 'scissors'):
#                         send_signals(classes[int(labels[i])], self.threshold, 'Tool',
#                                      datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
#
#                     if classes[int(labels[i])] in ('cougar', 'dalmatian', 'elephant', 'leopards', 'scorpion'):
#                         send_signals(classes[int(labels[i])], self.threshold, 'Animal',
#                                      datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
#                 except BaseException as err:
#                     print(f"Unexpected {err=}, {type(err)=}")
#         return frame
#
#     def get_frame(self):
#         """
#         This function is called when class is executed, it runs the loop to read the video frame by frame,
#         and write the output into a new file.
#         :return: void
#         """
#         while self._running:
#             start_time = time.time()
#             ret, orgi_frame = self.cap.read()
#             try:
#                 assert ret
#             except RuntimeError as er:
#                 print(er)
#             orgi_frame = cv2.resize(orgi_frame, (self.img_size, self.img_size))
#             ######################################################################################################
#             # Fourth, Check Tool & Animals: (Tools Animals Detection)
#             print("Fourth Stage starting... Tools Animals Detection")
#             orgi_frame = self.plot_boxes(tool_model, orgi_frame, tool_classes)
#             print(f'4- Tools Detection Ending... Total Time:{time.time() - start_time} s')
#
#             end_time = time.time()
#             fps = 1 / np.round(end_time - start_time, 2)
#             # print(f"Frames Per Second : {fps}")
#             # try:
#             #     fps1 = int(fps)
#             # except BaseException as err:
#             #     print(f"Unexpected {err=}, {type(err)=}")
#             #     fps1 = fps
#             print(f'Total Frame Rates: {fps}')
#             # cv2.putText(frame_resized, f'FPS: {fps1}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


first = Thread(target=stream)
first.start()

second = Thread(target=Faces(face_dataset_path=r"/home/abood/last/Databases",
                             blacklist_names=[],
                             visitor_names=[],
                             owner_names=[],
                             pets_names=[],
                             cap=capture).get_frame())
second.start()


# fourth = Thread(target=Tool(cap=capture,
#                             confidence=0.5,
#                             img_size=416
#                             ).get_frame())
# fourth.start()
