import cv2
import pickle
import argparse

import numpy as np
import pandas as pd

from time import time

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
deo = cv2.VideoWriter('./video/result.avi', fourcc, 30, (720, 720))


def process_image(image):
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = np.array(image).flatten()
    image = pd.DataFrame([image])

    return image


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', help='path to video')
args = vars(parser.parse_args())

sperm_mapping = {'0': 'Normal',
                 '1': 'Two heads',
                 '2': 'Two tails'}

video = cv2.VideoCapture(args['video'])
model = pickle.load(open('./model/knn.pkl', 'rb'))

counter = 0

while(video.isOpened()):
    ret, frame = video.read()

    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret1, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, _, dfg = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in _:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect

        crop = thresh[y:y+100, x:x+100]
        crop = np.array(crop).flatten()
        crop = pd.DataFrame([crop])

        cv2.rectangle(frame, (x, y), (x +100, y+100), (0, 255, 0), 2)
        cv2.putText(frame, list(map(lambda x: sperm_mapping.get(x), model.predict(crop)))[0], (x, y -10), 0, 0.3, (0, 255, 0))

    print counter
    counter += 1
    deo.write(frame)

deo.release()

# start_time = time()
#
# model = pickle.load(open('./model/gradient_descent.pkl', 'rb'))
#
# image = cv2.imread(args['image'], 0)
# image = process_image(image)
#
# print ('Test data were prepared in {} seconds.'.format(time() - start_time))
# print ('{} sperm was predicted in {} seconds.'.format(list(map(lambda x: sperm_mapping.get(x),
#                                                                model.predict(image)))[0], time() - start_time))
