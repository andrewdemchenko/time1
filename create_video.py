import os
import cv2

s_img1 = cv2.imread('./test/normal/1.png')
s_img2 = cv2.imread('./test/two_heads/6.png')
s_img3 = cv2.imread('./test/two_tails/11.png')


l_img = cv2.imread("./data/export.png")

height, width,layer = s_img1.shape

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter('./video/video.avi', fourcc, 10, (720, 720))

for i, j, k in zip(range(0, 420), range(-420, 0), range(-420, 0)):
    y_offset, x_offset = i, i
    y_offset1, x_offset1 = 500, -j+150
    y_offset2, x_offset2 = 80, -k

    M1 = cv2.getRotationMatrix2D((height / 2, width / 2), i, 1)
    dst1 = cv2.warpAffine(s_img1, M1, (height, width))
    M2 = cv2.getRotationMatrix2D((height / 2, width / 2), j, 1)
    dst2 = cv2.warpAffine(s_img2, M2, (height, width))
    M3 = cv2.getRotationMatrix2D((height / 2, width / 2), k, 1)
    dst3 = cv2.warpAffine(s_img3, M3, (height, width))

    copy = l_img.copy()

    copy[y_offset:y_offset + dst1.shape[0], x_offset:x_offset + dst1.shape[1]] = dst1
    copy[y_offset1:y_offset1 + dst1.shape[0], x_offset1:x_offset1 + dst1.shape[1]] = dst2
    copy[y_offset2:y_offset2 + dst1.shape[0], x_offset2:x_offset2 + dst1.shape[1]] = dst3

    video.write(copy)

video.release()
