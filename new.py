import os
import cv2
import pickle



import argparse

import numpy as np


from time import time
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

def load_images_from_folder(folder):
    images = []
    label = []

    for filename in os.listdir(folder):
        file_name = int(os.path.splitext(filename)[0])


        if file_name <= 11520:
            file_name = 1
        elif 11520 < file_name:
            file_name = 2

        image = cv2.imread(os.path.join(folder, filename), 0)
        ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        if image is not None:
            images.append(image.flatten())
            label.append(file_name)



    return images, label


start_time = time()


data, label = load_images_from_folder('./data/dataset/')
prepare_data_time = time()


# model1 = NearestCentroid()
# model1.fit(data, label)
# NearestCentroid(metric='manhattan', shrink_threshold=None)
# pickle.dump(model1, open('./model/knn_norm_manhetten.pkl', 'wb'))n_neighbors=3,
#                                                    weights='distance',
#                                                    algorithm='kd_tree',
#                                                    leaf_size=30, p=2,
#                                                    metric='minkowski',
#                                                    metric_params=None,
#                                                    n_jobs=-1)

model1 = KNeighborsClassifier()
model1.fit(data, label)
# pickle.dump(model1, open('./model/knnn.pkl', 'wb'))


# model11 = NearestCentroid()
# model11.fit(standardized_X, label)
# NearestCentroid(metric='euclidean', shrink_threshold=None)
# pickle.dump(model11, open('./model/knn_standart.pkl', 'wb'))
#
#
# model2 = SGDClassifier(alpha=0.00001, average=False, class_weight=None, epsilon=0.1,
#                       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#                       learning_rate='optimal', loss='hinge', max_iter=50, n_iter=None,
#                       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
#                       shuffle=True, tol=None, verbose=0, warm_start=False)
#
# model2.fit(normalized_X, label)
# pickle.dump(model2, open('./model/gradient_norm.pkl', 'wb'))
#
# model21 = SGDClassifier(alpha=0.00001, average=False, class_weight=None, epsilon=0.1,
#                       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#                       learning_rate='optimal', loss='hinge', max_iter=50, n_iter=None,
#                       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
#                       shuffle=True, tol=None, verbose=0, warm_start=False)
#
# model21.fit(standardized_X, label)
# pickle.dump(model21, open('./model/gradient_standart.pkl', 'wb'))
#
#
# model3 = LogisticRegression()
# model3.fit(normalized_X, label)
# pickle.dump(model3, open('./model/lr_norm.pkl', 'wb'))
#
#
# model31 = LogisticRegression()
# model31.fit(standardized_X, label)
# pickle.dump(model31, open('./model/lr_standart.pkl', 'wb'))


train_model_time = time()

print ('Train data were prepared in {} seconds.'.format(prepare_data_time - start_time))
print ('Model was prepared in {} seconds.'.format(train_model_time - prepare_data_time))










fourcc = cv2.VideoWriter_fourcc(*'MJPG')
deo = cv2.VideoWriter('./video/knnn.avi', fourcc, 10, (720, 720))

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', help='path to video')
args = vars(parser.parse_args())
#
sperm_mapping = {1: 'Good',
                 2: 'Bad'}

video = cv2.VideoCapture(args['video'])
# model = pickle.load(open('./model/knnn.pkl', 'rb'))

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

            print model1.predict([crop])

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

