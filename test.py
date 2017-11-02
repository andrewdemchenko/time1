import cv2
import pickle
import argparse

import numpy as np


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
deo = cv2.VideoWriter('./video/knnn.avi', fourcc, 10, (720, 720))

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', help='path to video')
args = vars(parser.parse_args())
#
sperm_mapping = {1: 'Good',
                 2: 'Bad'}

video = cv2.VideoCapture(args['video'])
model = pickle.load(open('./model/knnn.pkl', 'rb'))

counter = 0

while(video.isOpened()):
    ret, frame = video.read()
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret1, thresh = cv2.threshold(imgray, 127, 255, 0)

    contours, hierarchy, _ = cv2.findContours(thresh, 1, 2)
    for c in hierarchy:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect

        if len(c) > 20:
            crop = thresh[y-20 :y + 80, x-20 :x + 80]
            crop = np.array(crop).flatten()

            print model.predict([crop])

            # label = list(map(lambda x: sperm_mapping.get(x), model.predict([crop])))[0]

            # try:
            #     label = list(map(lambda x: sperm_mapping.get(x), model.predict([crop])))[0]
            # except Exception:
            #     label = '1'

            if label == 'Good':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), 0, 0.3, (0, 255, 0))
            elif label == 'Bad':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, label, (x, y - 10), 0, 0.3, (0, 0, 255))
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 255, 255), 2)
                cv2.putText(frame, 'Undefined', (x, y - 10), 0, 0.3, (255, 255, 255))
        else:
            pass

    print counter
    counter += 1

    deo.write(frame)

deo.release()

