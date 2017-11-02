import cv2
import pickle
import argparse

import numpy as np
import pandas as pd

from time import time


def process_image(image):
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = np.array(image).flatten()
    image = pd.DataFrame([image])

    return image

sperm_mapping = {'0': 'Normal',
                 '1': 'Two heads',
                 '2': 'Two tails'}

start_time = time()

model = pickle.load(open('./model/knn.pkl', 'rb'))

image = cv2.imread("./test/normal/1.png", 0)
# height, width = image.shape
# M1 = cv2.getRotationMatrix2D((height / 2, width / 2), 240, 1)
# dst1 = cv2.warpAffine(image, M1, (height, width))
cv2.imshow("lol",image)
cv2.waitKey(0)
image = process_image(image)

print ('Test data were prepared in {} seconds.'.format(time() - start_time))
print ('{} sperm was predicted in {} seconds.'.format(list(map(lambda x: sperm_mapping.get(x),
                                                               model.predict(image)))[0], time() - start_time))

# start_time = time()
#
# model = pickle.load(open('./model/knn.pkl', 'rb'))
#
# image = cv2.imread('1.jpg', 0)
# ret1, thresh = cv2.threshold(image, 127, 255, 0)
# thresh = np.array(thresh).flatten()
#
# print model.predict([thresh])[0]
#
# print ('Test data were prepared in {} seconds.'.format(time() - start_time))
# print ('{} sperm was predicted in {} seconds.'.format(list(map(lambda x: sperm_mapping.get(x),
#                                                                model.predict([thresh])))[0], time() - start_time))
