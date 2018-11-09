import cv2 as cv
import numpy as np


viedo_list =['D:/bot_opencv/dectect/dectect/image/4.mp4'
    ,'D:/bot_opencv/dectect/dectect/image/5.mp4'
    ,'D:/bot_opencv/dectect/dectect/image/6.mp4'
    ,'D:/bot_opencv/dectect/dectect/image/n3.mp4'
    ,'D:/bot_opencv/dectect/dectect/image/n1.mp4']

cap0 = cv.VideoCapture(viedo_list[0])
cap1 = cv.VideoCapture(viedo_list[1])
cap2 = cv.VideoCapture(viedo_list[2])
cap3 = cv.VideoCapture(viedo_list[3])
cap4 = cv.VideoCapture(viedo_list[4])

while True:
    ret0, fram0 = cap0.read()
    ret1, fram1 = cap1.read()
    ret2, fram2 = cap2.read()
    ret3, fram3 = cap3.read()
    ret4, fram4 = cap4.read()


    cv.imshow('viedo0', fram0)
    cv.imshow('viedo1', fram1)
    cv.imshow('viedo2', fram2)
    cv.imshow('viedo3', fram3)
    cv.imshow('viedo4', fram4)

    cv.waitKey(10)
