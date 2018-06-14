import cv2 as cv
import numpy as np
from matplotlib import  pyplot as plt

img=cv.imread('01.jpg')

rows,cols,ch=img.shape
pts1=np.float32([[50,50],[200,50],[50,200]])



pts1=np.float32([[10,100],[200,50],[100,250]])

M =cv.getAffineTransform(pts1,pts1)
dst=cv.warpAffine(img,M,(cols,rows))

cv.circle(img,(50,50),3,(255,0,0),-1)
cv.circle(img,(200,50),3,(255,0,0),-1)
cv.circle(img,(50,200),3,(255,0,0),-1)

cv.circle(dst,(10,100),3,(255,0,0),-1)
cv.circle(dst,(200,50),3,(255,0,0),-1)
cv.circle(dst,(100,250),3,(255,0,0),-1)

# plt.subplot(121,plt.imshow(img),plt.title('Input'))
# plt.subplot(121,plt.imshow(img),plt.title('Output'))
# plt.show()

cv.imshow('img',img)
cv.imshow('dst',dst)
cv.waitKey()


