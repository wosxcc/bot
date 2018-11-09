import cv2 as cv
import numpy as np
import datetime
cap =cv.VideoCapture('E:/Desk_Set/22.mp4')


while True:
    ret, img = cap.read()

    img =cv.resize(img,(400,400),interpolation=cv.INTER_CUBIC)
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    start_time =datetime.datetime.now()

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 3.0)
    K = 8
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    print("耗时",datetime.datetime.now()-start_time)
    cv.imshow('res2',res2)
    cv.waitKey(10)
