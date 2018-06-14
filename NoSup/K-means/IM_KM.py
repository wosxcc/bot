import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('1.jpg',0)
# img=cv.cvtColor(image,cv.COLOR_RGB2GRAY)

kernel= np.ones((5,5),np.uint8)


ret,th=cv.threshold(img,50,255,cv.THRESH_BINARY)

closing=cv.morphologyEx(th,cv.MORPH_OPEN,kernel)

opening=cv.morphologyEx(closing,cv.MORPH_CLOSE,kernel)
dilating=cv.dilate(opening,kernel,iterations=3)

erosion=cv.erode(dilating,kernel,iterations=3)

image ,contours,hierarchy=cv.findContours(erosion,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

img_k =cv.drawContours(img,contours,-1,(255),3)
print(contours[0].shape)
xx=contours[0].reshape(contours[0].shape[0],2)

arg_x=int(sum(xx[:,0])/contours[0].shape[0])
arg_y=int(sum(xx[:,1])/contours[0].shape[0])

print('arg_x',arg_x)
print('arg_y',arg_y)

cv.circle(img,(arg_x,arg_y),3,(0),-1)
# print(xx)
cv.imshow('img', img_k)
# cv.imshow('opening',opening)
# cv.imshow('closing',closing)

cv.imshow('dilating',erosion)
cv.imshow('img_k',img_k)








cv.waitKey()
cv.destroyAllWindows()