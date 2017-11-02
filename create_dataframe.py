import os
import cv2


data_prepared = './test/data_prepared/'


l_img = cv2.imread("./data/large.png")

counter = 1

dataset = os.listdir(data_prepared)
dataset = sorted([int(os.path.splitext(i)[0]) for i in dataset])
dataset = [str(i) + '.png' for i in dataset]

for a in dataset:
    a = cv2.imread(data_prepared + a)
    rows, cols, chanel = a.shape
    for i in range(0,80,10):
        for j in range(0,80,10):
            for k in range(0,360,10):
                y_offset, x_offset = i, j
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2),k, 1)
                dst = cv2.warpAffine(a, M, (cols, rows))
                copy = l_img.copy()
                copy[y_offset : y_offset + dst.shape[0], x_offset : x_offset + dst.shape[1]] = dst
                cv2.imwrite('./data/dataset/{}.png'.format(counter), copy)

                counter += 1
                print counter










# for degree in range(1, 360):
#     M = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
#     dst = cv2.warpAffine(good, M, (cols, rows))
#
#     M1 = cv2.getRotationMatrix2D((cols1 / 2, rows1 / 2), degree, 1)
#     dst1 = cv2.warpAffine(bad1, M1, (cols1, rows1))
#
#     M2 = cv2.getRotationMatrix2D((cols2 / 2, rows2 / 2), degree, 1)
#     dst2 = cv2.warpAffine(bad2, M2, (cols2, rows2))
#
#     cv2.imwrite('./train/normal/{}.png'.format(degree), dst)
#     cv2.imwrite('./train/two_heads/{}.png'.format(degree), dst1)
#     cv2.imwrite('./train/two_tails/{}.png'.format(degree), dst2)

# good = cv2.imread('1.png')
# bad1 = cv2.imread('3.png')
# bad2 = cv2.imread('5.png')
#
# height, width, layers = good.shape














