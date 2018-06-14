import numpy as np
from  scipy.cluster.vq import *
from  scipy.misc import  imresize
import cv2 as cv
import datetime
img = cv.imread('4.jpg')
xtime=datetime.datetime.now()
imgs=cv.resize(img,(480,480),cv.INTER_CUBIC)
steps= 20
sx=int(imgs.shape[1]/steps)
sy=int(imgs.shape[0]/steps)

fmeans =[]
for x in range(sx):
    for y in range(sy):
        R = np.mean(imgs[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps, 0])
        G = np.mean(imgs[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps, 1])
        B = np.mean(imgs[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps, 2])
        fmeans.append([R,G,B])
fmeans =np.array(fmeans,'f')
# print('fmeans',fmeans)
centroids,variance=kmeans(fmeans,2)

print(centroids,variance)


code,distance=vq(fmeans,centroids)
print(code,distance)
codess=np.reshape(code,(sx,sy))
# print(codess.shape)
nimg=np.zeros((480,480,3),np.uint8)
# print(codess[0][0])
for x in range(sx):
    for y in range(sy):
        if codess[x][y]==0:
            nimg[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps, 0] = int(centroids[0][0])
        elif codess[x][y]==1:
            nimg[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps, 1] = int(centroids[1][1])
        elif codess[x][y] == 2:
            nimg[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps, 2] = int(centroids[2][2])
        elif codess[x][y] == 3:
            nimg[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps, :] = int(centroids[3])    #[255, 255, 0]
        # elif codess[x][y] == 4:
        #     nimg[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps, :] = int(centroids[0])
        elif codess[x][y] == 5:
            nimg[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps, :] = [0, 255, 255]
        else:

            nimg[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps,:] = [0, 0, 0]
# print(centroids)

print('耗时:',datetime.datetime.now()-xtime)

cv.imshow('imgs',imgs)
cv.imshow('nimg',nimg)
cv.waitKey()


# print('variance',variance)
#
# print('code',code)
# print('均值：',sum(distance)/len(distance))
# print('distance',distance)
# codeim=code.resize(steps,steps)
# print('codeim',codeim)
# # codeim= imresize(codeim,img.shape[:2],interp='nearest')
#
#
# cv.imshow('imgs',imgs)
# cv.imshow('codeim',codeim)
cv.waitKey()
cv.destroyAllWindows()

