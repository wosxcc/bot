import cv2 as cv
from MY_Function.cv_function import *


kernel =[[1,-1,1],
         [1,-1,1],
         [1,-1,1],]

img =cv.imread('50002.jpg')
cv.imshow('img',img)
# img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img =dram_chinese(img,'舟山高数',50,50,40,(255,255,0))
cv.imshow('imgdd',img)
imgss =img_rotate(img, angle =20)
cv.imshow('imgss',imgss)
cv.waitKey()