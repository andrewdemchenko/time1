import os
import cv2

s_img1 = cv2.imread("small.png")
s_img2 = cv2.imread("3.png")
s_img3 = cv2.imread("5.png")
photos_for_video = './video/photos/'

l_img = cv2.imread("large.png")

height, width, layers = s_img1.shape

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter('./video/video.avi', fourcc, 30, (720, 720))

# M1 = cv2.getRotationMatrix2D((height / 2, width / 2), 240, 1)
# dst1 = cv2.warpAffine(s_img1, M1, (height, width))
# M2 = cv2.getRotationMatrix2D((height / 2, width / 2), 25, 1)
# dst2 = cv2.warpAffine(s_img2, M2, (height, width))
# M3 = cv2.getRotationMatrix2D((height / 2, width / 2), 90, 1)
# dst3 = cv2.warpAffine(s_img3, M3, (height, width))



rotate_plus = []
rotate_minus = []

for k in range(20):
    rotate_plus.append(k)
    rotate_minus.append(-k)

for i, j, k in zip(range(0, 420), range(-420, 0), range(-420, 0)):
    y_offset, x_offset = i, i
    y_offset1, x_offset1 = 500, -j+150
    y_offset2, x_offset2 = 80, -k

    # M1 = cv2.getRotationMatrix2D((height / 2, width / 2), i, 1)
    # dst1 = cv2.warpAffine(s_img1, M1, (height, width))
    # M2 = cv2.getRotationMatrix2D((height / 2, width / 2), j, 1)
    # dst2 = cv2.warpAffine(s_img2, M2, (height, width))
    # M3 = cv2.getRotationMatrix2D((height / 2, width / 2), k, 1)
    # dst3 = cv2.warpAffine(s_img3, M3, (height, width))

    copy = l_img.copy()

    copy[y_offset:y_offset + s_img1.shape[0], x_offset:x_offset + s_img1.shape[1]] = s_img1
    copy[y_offset1:y_offset1 + s_img1.shape[0], x_offset1:x_offset1 + s_img1.shape[1]] = s_img2
    copy[y_offset2:y_offset2 + s_img1.shape[0], x_offset2:x_offset2 + s_img1.shape[1]] = s_img3

    video.write(copy)

video.release()
