import os
import cv2
import pickle

import numpy as np
import pandas as pd

from time import time
from sklearn.linear_model import SGDClassifier


def process_folder(folders):
    images = []

    for folder in folders:
        for filename in os.listdir(folders.get(folder)):
            image = cv2.imread(os.path.join(folders.get(folder), filename), 0)
            ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

            if image is not None:
                images.append(np.append(image, folder).flatten())

    images = pd.DataFrame(images)
    data, label = images.ix[:, :9999], np.ravel(images.ix[:, 10000:])

    return data, label


normal = './train/normal/'
two_tails = './train/two_tails/'
two_heads = './train/two_heads/'

start_time = time()

data, label = process_folder({'0': normal,
                              '1': two_heads,
                              '2': two_tails})

prepare_data_time = time()

model = SGDClassifier(alpha=0.00001, average=False, class_weight=None, epsilon=0.1,
                      eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                      learning_rate='optimal', loss='hinge', max_iter=50, n_iter=None,
                      n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
                      shuffle=True, tol=None, verbose=0, warm_start=False)

model.fit(data, label)
pickle.dump(model, open('./model/gradient_descent.pkl', 'wb'))

train_model_time = time()

print ('Train data were prepared in {} seconds.'.format(prepare_data_time - start_time))
print ('Model was prepared in {} seconds.'.format(train_model_time - prepare_data_time))
