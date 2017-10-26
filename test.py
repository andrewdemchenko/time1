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


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='path to image')
args = vars(parser.parse_args())

sperm_mapping = {'0': 'Normal',
                 '1': 'Two heads',
                 '2': 'Two tails'}

start_time = time()

model = pickle.load(open('./model/gradient_descent.pkl', 'rb'))

image = cv2.imread(args['image'], 0)
image = process_image(image)

print ('Test data were prepared in {} seconds.'.format(time() - start_time))
print ('{} sperm was predicted in {} seconds.'.format(list(map(lambda x: sperm_mapping.get(x),
                                                               model.predict(image)))[0], time() - start_time))
