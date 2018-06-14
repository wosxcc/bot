import cv2 as cv
import numpy as np
from  scipy.cluster.vq import *
from  scipy.misc import  imresize
img =cv.imread('2.jpg')

imgs=cv.resize(img,(480,480),cv.INTER_CUBIC)
steps= 32
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
print('fmeans',fmeans)
centroids,variance=kmeans(fmeans,4)
code,distance=vq(fmeans,centroids)
print('variance',variance)

print('code',code)
print('均值：',sum(distance)/len(distance))
print('distance',distance)
codeim=code.resize(steps,steps)
print('codeim',codeim)
# codeim= imresize(codeim,img.shape[:2],interp='nearest')


cv.imshow('imgs',imgs)
cv.imshow('codeim',codeim)
cv.waitKey()
cv.destroyAllWindows()

