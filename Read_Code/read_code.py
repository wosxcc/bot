import cv2 as cv
import os
import numpy as np

path= 'H:/Chrome_drown/caffe_verify_code/train_notsplit'
for file in os.listdir(path):
    img =cv.imread(path+'/'+file,0)
    cv.imshow('img', img)
    img=cv.resize(img,None, fx=2.0,fy=2.0, interpolation=cv.INTER_CUBIC)
    ret ,img2 =cv.threshold(img,220,255,cv.THRESH_BINARY_INV) ##二值化

    img2=cv.medianBlur(img2,5)   ##中值滤波
    kernel=np.ones((3,3),np.uint8)
    img2 = cv.erode(img2, kernel, iterations=1)   ##腐蚀
    img2 = cv.dilate(img2, kernel, iterations=1)  ##膨胀
    # img2 = cv.medianBlur(img2, 7)  ##中值滤波
    # loaplacian=cv.Laplacian(img2,cv.CV_64F)
    # img2=cv.Sobel(img2,cv.CV_64F,1,0,ksize=5)      ##X方向卷积
    # img2 = cv.dilate(img2, kernel, iterations=1)   ##膨胀
    # img2 = cv.erode(img2, kernel, iterations=1)
    # img2 = cv.dilate(img2, kernel, iterations=1)
    # # for i in range(5):
    # #     img2=cv.erode(img2,kernel,iterations=1)
    # #     img2=cv.dilate(img2,kernel,iterations=1)
    # # cv.imshow('img2', img2)
    cv.imshow(file, img2)
    cv.waitKey()
    cv.destroyAllWindows()



