import numpy as np
import cv2
import time
import datetime

cap = cv2.VideoCapture('E:/BOT_Car/bot_car/test2.mp4')

kernelss=np.ones((2,2),np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2()     # 创建背景减法器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame1 = np.zeros((640, 480))
out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.avi', fourcc, 5.0,
                      np.shape(frame1))

while (1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)


    # fgmask = cv2.erode(fgmask, kernelss, iterations=1)
    # fgmask = cv2.erode(fgmask, kernelss, iterations=1)
    ret, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
    fgmask = cv2.medianBlur(fgmask, 5)
    # fgmask = cv2.dilate(fgmask, kernelss, iterations=1)
    # fgmask = cv2.dilate(fgmask, kernelss, iterations=1)
    fgmask =cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernelss)
    ret, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
    cv2.imshow('fgmask',fgmask)
    (_, cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #寻找图像轮廓
    maxArea = 0
    xtime =datetime.datetime.now()
    print('里面存放什么东西',cnts)
    print(cnts[0 ].shape)
    for c in cnts:
        Area = cv2.contourArea(c) #计算轮廓面积
        if Area < maxArea:
            # if cv2.contourArea(c) < 500:
            (x, y, w, h) = (0, 0, 0, 0)
            continue
        else:
            if Area < 180:
                (x, y, w, h) = (0, 0, 0, 0)
                continue
            else:
                maxArea = Area
                m = c   # m为存放的一组坐标
                print('看看m是什么',m)
                (x, y, w, h) = cv2.boundingRect(m) #获取最小外接矩形边框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out.write(frame)
    print('耗时:', datetime.datetime.now() - xtime)
    cv2.imshow('frame', frame)
    cv2.waitKey()
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
out.release()
cap.release()
cv2.destoryAllWindows()