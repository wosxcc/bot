import matplotlib.pyplot as plt
import numpy as np
from  scipy.cluster.vq import *
import cv2 as cv
from sklearn.datasets.samples_generator import make_blobs

img=cv.imread('2.jpg')
# cv.imshow('img',img)
img=cv.resize(img,(480,480),cv.INTER_CUBIC)
imgs=np.float32(img)
# print(imgg.shape)
# cv.waitKey()


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



centroids,variance = kmeans(fmeans,2)
code,distance=vq(fmeans,centroids)


nimg=np.zeros((480,480,3),np.uint8)
for x in range(sx):
    for y in range(sy):
        if code[x*y]==1:
            nimg[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps, 0] = 255

        else:
            nimg[y * steps:(y + 1) * steps, x * steps:(x + 1) * steps, 2] = 255
print(centroids)
cv.imshow('nimg',nimg)
cv.waitKey()
print(code)
print('distance',distance)


