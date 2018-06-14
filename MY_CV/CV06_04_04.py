###相似图片搜索
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

min_count=10


img1=cv.imread('P13.jpg')
# sift = cv.SIFT()  ##cv.xfeatures2d.SIFT_create()
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
FLANN_INDEX_KDTREE = 0
indesx_params= dict(algorithm=FLANN_INDEX_KDTREE,trees=5)    ##设置索引参数
search_params =dict(checks = 50)                             ##设置查找参数
flann= cv.FlannBasedMatcher(indesx_params,search_params)


matching_line = {}
for file in os.listdir('./image'):
    if file[-4:]=='.jpg':
        # print('./image'+file)
        img2=cv.imread('./image/'+file)
        kp2, des2 = sift.detectAndCompute(img2, None)
        matches = flann.knnMatch(des1,des2,k=2)
        good =[]
        for m,n in matches:
            if m.distance <0.7*n.distance:
                good.append(m)

        matching_line[file]=len(good)
        # print(len(good))


print(matching_line)
hot_img=sorted(matching_line, key=lambda m:matching_line[m],reverse=True)[:10]


print(hot_img)



        # cv.imshow('img1',img1)
        # cv.imshow('img2',img2)
        # cv.waitKey()

