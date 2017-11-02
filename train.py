import os
import cv2
import pickle


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


        if file_name <= 10000:
            file_name = 1
        elif 11520 < file_name <= 21520:
            file_name = 2
	else:
	    continue

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
# pickle.dump(model1, open('./model/knn_norm_manhetten.pkl', 'wb'))

model1 = KNeighborsClassifier(n_neighbors=3,weights='distance',
                                                   algorithm='kd_tree',
                                                   leaf_size=30, p=2,
                                                   metric='minkowski',
                                                   metric_params=None,
                                                   n_jobs=-1)
model1.fit(data, label)
pickle.dump(model1, open('./model/knnn.pkl', 'wb'))


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








