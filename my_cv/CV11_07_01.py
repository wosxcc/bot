import  numpy as np
import cv2 as cv



cap =cv.VideoCapture('E:/Desk_Set/22.mp4')


while True:
    ret, fram = cap.read()
    gray = cv.cvtColor(fram, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    res = np.hstack((centroids, corners))
    # np.int0 可以用来省略小数点后儑的数字虹儍四㮼五入虺。
    res = np.int0(res)
    print(res)

    for list in res:
        cv.circle(fram,(list[0],list[1]),3,(0, 0, 255),-1)
        cv.circle(fram, (list[2], list[3]), 3, (0, 255, 0), -1)

    # fram[res[:, 1], res[:, 0]] = [0, 0, 255]
    # fram[res[:, 3], res[:, 2]] = [0, 255, 0]


    cv.imshow('img',fram)
    cv.waitKey(10)