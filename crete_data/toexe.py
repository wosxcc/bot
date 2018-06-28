# -*- encoding=utf-8 -*-

import cv2 as cv


import ctypes

whnd = ctypes.windll.kernel32.GetConsoleWindow()
if whnd != 0:
    ctypes.windll.user32.ShowWindow(whnd, 0)
    ctypes.windll.kernel32.CloseHandle(whnd)


cap =cv.VideoCapture(0)

while(1):
    ret ,img =cap.read()

    cv.imshow('img',img)
    cv.waitKey(10)
# ;D:\Python