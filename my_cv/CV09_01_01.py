import cv2 as cv
import numpy as np

img = cv.imread('4.jpg')

img = cv.resize(img,(200,200),interpolation=cv.INTER_CUBIC)
# img_gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#
# ret, fgmask = cv.threshold(img_gray, 250, 255, cv.THRESH_BINARY_INV)
#
# cv.imshow('fgmask',fgmask)
height,weight,ch =img.shape

# for x in range(weight):
#     for y in range(height):
#         if img[x][y][0] == 255 and  img[x][y][1] == 255 and img[x][y][2] == 255:
#             img[x][y]=[0,0,0]





cv.imshow('img',img)
cv.waitKey()


for i in range((weight-50)//10):
    for j  in range((height-50)//10):
        pts1 = np.float32([[i*10,j*10],[50,50],[height,weight]])
        pts2 = np.float32([[i*10+50,j*10+50],[50,50],[height-10,weight-50]])
        m= cv.getAffineTransform(pts1,pts2)
        dst=cv.warpAffine(img,m,(height,weight))
        cv.imshow('dst',dst)
        cv.waitKey(200)

# cv.imshow('imgss',img)
# cv.imshow('dst',dst)
cv.waitKey()


