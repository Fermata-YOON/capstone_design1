# 카메라 연결 import
import cv2


import numpy as np
#모델 불러올 것
from keras.models import load_model
from statistics import mode
import tensorflow as tf


from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

#시리얼 통신 라이브러리
import serial
import serial.tools.list_ports
import re
import time

total = 0

USE_WEBCAM = True # If false, loads video file source


# parameters for loading data and images
#모델 경로
emotion_model_path = './models/emotion_model.hdf5'
#모델 Data
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
# 모델 불러오기
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

#tensorflow lite로 경량화 시키기.
#converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(emotion_classifier) # path to the SavedModel directory
#emotion_classifier = converter
#tflite_model = converter.convert()

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
# 얼굴인식 화면 세팅
emotion_window = []

# starting video streaming

#컴퓨터 카메라 키기
cv2.namedWindow('window_frame')
#동영상 파일의 경로를 파라미터로 받으며 0으로 입력 시 컴퓨터에 연결된 카메라를 스트리밍으로 보여준다.
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source

# 시리얼 포트 연결
ports = serial.tools.list_ports.comports()

available_port = []

for p in ports:
    available_port.append(p.device)

print(available_port)


#포트번호 설정
CMD = serial.Serial(
    port='/dev/cu.usbserial-1130',
    baudrate="115200"
)

count = 0

#카메라 구동 & 자동차 통신
while cap.isOpened(): # True:
    #cv2.read와 다르게 cap에 있는 데이터, 즉 영상의 데이터를 하나씩 불러온다.
    ret, bgr_image = cap.read()

    #bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        #감정인식 예측
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue


        #total이란 우리가 이값을 stm32에 보내서 제동을 걸지말지 결정하는 변수
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))

            total = 1

        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))

            total = 2

        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))

            total = 3

        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
            count += 1
            if count > 4:
                count = 0
                total = 4

        else:
            color = emotion_probability * np.asarray((0, 255, 0))
            total = 0


        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)

    print(total)

    total = str(total)

    CMD.write(total.encode())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#자원 반납, 즉 메모리 해제
cap.release()

cv2.destroyAllWindows()
