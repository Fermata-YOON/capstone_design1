#라즈베리 파이와  pc간의 직렬통신
import serial
from time import sleep
import time
import re

port1 = "/dev/ttyS0"
emotion = serial.Serial(port1,baudrate = 115200)

port2 = "/dev/ttyACM0"
arduino = serial.Serial(port2,baudrate = 115200)
arduino.flushInput()

def rasberry_aduino(svm_clf):
    global ras, adu

    while True:
        arduinoinput = arduino.readline()
        adu = arduinoinput.decode()
        #serialFromArduino.write(test.encode())
        total = emotion.read()
        ras = total.decode()
        #print(total.decode(),arduinoinput.decode())
        #print()
        adu = re.sub(r'[^0-9]', '', adu)
        if(len(adu) != 0):
            adu = int(adu)
            ras = int(ras)
            li = [adu,ras]
            if len(li) >= 2:
                
                #print(li)
                print(li)
                play(adu, ras, svm_clf)
            else:
                print(0)
        



##################################################################################
#svm 모델 코드
import numpy as np
from sklearn import svm
def play(a,b, model):
    svm = model
    test_data = np.array([[a,b]])
    label = svm.predict(test_data)
    label = str(label)
    arduino.write(label.encode())
    print(label)
def model():
    x_train = np.array([[2, 0], [4, 3], [6, 1], [8, 3],[10, 0],[12, 1], [14, 2], [16, 3], [18, 0], [20, 1],
                    [22, 2],[24, 2],[26, 0], [28, 3],[30, 2],[32, 3],[34, 0], [36, 1],[38, 2], [40, 3],
                    [42, 0], [44, 2],[46, 2], [48, 3],[47, 0],

                    [50, 1], [53, 2],[55, 3], [57, 1], [59, 3], [61, 2], [63, 0], [65, 2], [67, 1], [69, 3],
                    [71, 2], [73, 3],[75, 2], [77, 2], [79, 2], [81, 2], [83, 1], [85, 3], [87, 2], [89, 2],
                    [91, 3], [93, 1],[95, 0], [97, 3], [99, 1],

                    [70, 4], [72, 4],[68, 4], [64, 4], [65, 4], [61, 4], [63, 4], [65, 4], [67, 4], [69, 4],
                    [71, 4], [73, 4],[75, 4], [77, 4], [79, 4], [81, 4], [83, 4], [85, 4], [87, 4], [89, 4],
                    [91, 4], [93, 4],[95, 4], [97, 4], [99, 4]])
    y_train = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

    # SVM생성
    svm_clf = svm.SVC(kernel='linear')
    # 학습
    svm_clf.fit(x_train, y_train)
    return svm_clf
 ##################################################################################
#실행코드
svm_clf = model()
rasberry_aduino(svm_clf)
