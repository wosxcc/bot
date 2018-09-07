import cv2 as cv
import numpy as np


img1 = cv.imread('./Desk/184.jpg')

img1 =cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img1 =cv.resize(img1,(600,600),cv.INTER_CUBIC)
img2 = cv.imread('./Desk/170.jpg')
img2 =cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
img2 =cv.resize(img2,(600,600),cv.INTER_CUBIC)

cv.imshow('img1',img1)

addimg =cv.bitwise_and(img2,img1)
cv.imshow('addimg',addimg)

orimg =cv.bitwise_or(img2,img1)
cv.imshow('orimg',orimg)

xorimg =cv.bitwise_xor(img2,img1)
cv.imshow('xorimg',xorimg)

notimg =cv.bitwise_not(img2,img1)
cv.imshow('notimg',notimg)




cv.waitKey()


