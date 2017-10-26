import cv2

# good = cv2.imread('good.png', 0)
# rows,cols = good.shape
#
# bad1 = cv2.imread('bad.png', 0)
# rows1,cols1 = bad1.shape

bad2 = cv2.imread('bad4.png',0)
rows2,cols2 = bad2.shape

for degree in range(1, 360):
    # M = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
    # dst = cv2.warpAffine(good, M, (cols, rows))
    #
    # M1 = cv2.getRotationMatrix2D((cols1 / 2, rows1 / 2), degree, 1)
    # dst1 = cv2.warpAffine(bad1, M1, (cols1, rows1))

    M2 = cv2.getRotationMatrix2D((cols2 / 2, rows2 / 2), degree, 1)
    dst2 = cv2.warpAffine(bad2, M2, (cols2, rows2))

    # cv2.imwrite('./bad1/{}.png'.format(degree), dst1)
    # cv2.imwrite('./good/{}.png'.format(degree), dst)
    cv2.imwrite('./train/two_heads/{}.png'.format(degree), dst2)