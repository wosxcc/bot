import cv2 as cv
from deeplab_v3.model import Deeplabv3
import numpy as np
from matplotlib import pylab as plt
import datetime


deeplab_model =Deeplabv3()


cap =cv.VideoCapture(0)
cap2=cv.VideoCapture('E:/xcc_download/viedo2.mp4')

while(1):
    ret, img =cap.read()
    ret2, img2=cap2.read()
    w,h ,_ =img.shape
    ratio= 512./np.max([w,h])
    resized = cv.resize(img, (int(ratio * h), int(ratio * w)))
    resized = resized / 127.5 - 1.
    pad_x = int(512 - resized.shape[0])
    resized2 = np.pad(resized, ((0, pad_x), (0, 0), (0, 0)), mode='constant')
    res = deeplab_model.predict(np.expand_dims(resized2, 0))
    labels = np.argmax(res.squeeze(), -1)

    # print(np.maximum(labels[:-pad_x], 255))
    # img = cv.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # imgg =cv.cvtColor(np.maximum(labels[:-pad_x], 255), cv.COLOR_GRAY2BGR)
    # cv.imshow('imagess',imgg)
    # cv.waitKey()
    # plt.imshow(labels)

    # maxzhi =max(labels[:-pad_x].astype(np.uint8))
    img_gray = labels[:-pad_x].astype(np.uint8)

    ret,th= cv.threshold(img_gray,10,255,cv.THRESH_BINARY)

    # img2 =cv.imread('P06.jpg')
    img2=cv.resize(img2,(labels[:-pad_x].shape[1],labels[:-pad_x].shape[0]),cv.INTER_CUBIC)
    img=cv.resize(img,(labels[:-pad_x].shape[1],labels[:-pad_x].shape[0]),cv.INTER_CUBIC)
    # print(img2.shape,img.shape)

    mask_inv=cv.bitwise_not(th)

    img_bg = cv.bitwise_and(img, img, mask=th)
    img_fg = cv.bitwise_and(img2, img2, mask=mask_inv)

    dstt=cv.add(img_bg,img_fg)
    cv.imshow('dstt',dstt)
    # cv.imshow('img_fg',img_fg)
    cv.waitKey(10)
