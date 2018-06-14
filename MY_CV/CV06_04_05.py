###图片自动裁剪
import cv2 as cv
import numpy as np

res_bian=20
imgx =cv.imread('1.jpg')
img =cv.cvtColor(imgx,cv.COLOR_BGR2GRAY)
res ,th=cv.threshold(img,215,255,cv.THRESH_BINARY_INV)

kernel = np.ones((3,3),np.uint8)
dil_img=cv.dilate(th,kernel,iterations=3)       ##膨胀
ero_img=cv.erode(dil_img,kernel,iterations=12)   ##腐蚀
dil_img2=cv.dilate(ero_img,kernel,iterations=6)       ##膨胀

image,contours,hierarchy = cv.findContours(dil_img2,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
xxc = np.reshape(contours[0],(contours[0].shape[0],contours[0].shape[2]))
minx=min(xxc[:,0])
miny=min(xxc[:,1])
maxx=max(xxc[:,0])
maxy=max(xxc[:,1])
if minx>=res_bian:
    minx=minx-res_bian
else:
    minx=0
if miny>=res_bian:
    miny=miny-res_bian
else:
    miny=0
if maxx+res_bian<=img.shape[1]:
    maxx=maxx+res_bian
else:
    maxx=img.shape[1]
if maxy+res_bian<=img.shape[0]:
    maxy=maxy+res_bian
else:
    maxy=img.shape[0]

ximg=imgx[miny:maxy,minx:maxx,:]
# cv.imshow('dil_img2',dil_img2)
cv.imshow('img',img)
# cv.imshow('th',th)
cv.imshow('ximg',ximg)
cv.waitKey()
