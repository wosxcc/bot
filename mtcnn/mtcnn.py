from mtcnn.mtcnn import MTCNN
import cv2 as cv
import datetime
import os

path='E:/Face_Hand/train'

for file_name in os.listdir(path):
    if file_name[-4:]=='.jpg':
        img=cv.imread(path+'/'+file_name)
        cv.imshow('img',img)
        cv.waitKey()
