import numpy as np
import cv2 as cv

img = cv.imread('01.jpg')
cv.imshow('yimg',img)

imgt = img[:,:,::-1].transpose((2,0,1))

print(imgt.T.shape)
cv.imshow('timf',imgt.T)
cv.waitKey()

imgf =  imgt[:,:,::-1].T.transpose((2,1,0))

print(imgt.shape)

print(imgf.shape)
cv.imshow('timf',imgf)


cv.waitKey()