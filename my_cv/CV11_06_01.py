import cv2 as cv
import numpy as np

cap =cv.VideoCapture('E:/Desk_Set/22.mp4')


while True:
    ret, fram = cap.read()

    gray_img = cv.cvtColor(fram,cv.COLOR_BGR2GRAY)
    gray_img =cv.blur(gray_img, (3, 3))
    canny_img = cv.Canny(gray_img,100,200)

    image,contours,hier = cv.findContours(canny_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    # for alist in contours:
    #     for blist in alist:
    #         for clist in blist:
    #             cv.circle(fram,(clist[0],clist[1]),2,(255,0,0),-1)

    # cv.drawContour(fram, contours, -1, (0, 255, 0), 3)
    cv.drawContours(fram,contours,-1,(255,0,255),3)
    # cv.drawContours(img, contours, -1, (0, 0, 255), 3)

    # print("轮廓",contours)
    print("关系",hier)
    cv.imshow("imgs",fram)
    cv.imshow("img",canny_img)
    cv.waitKey(10)
