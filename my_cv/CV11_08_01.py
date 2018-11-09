import cv2 as cv
import numpy as np
yimg1 = cv.imread('./Desk/184.jpg')
imgs_res =cv.resize(yimg1,(600,600),cv.INTER_CUBIC)
img1 =cv.cvtColor(imgs_res,cv.COLOR_BGR2GRAY)

yimg2 = cv.imread('./Desk/170.jpg')

img_res =cv.resize(yimg2,(600,600),cv.INTER_CUBIC)
img2 =cv.cvtColor(img_res,cv.COLOR_BGR2GRAY)

kernel = np.ones((3,3),np.uint8)
img1 =cv.blur(img1, (3, 3))
canny_img1 = cv.Canny(img1,120,200)
xorimg1 =cv.dilate(canny_img1,kernel,iterations = 1)

img2=cv.blur(img2, (3, 3))
canny_img2 = cv.Canny(img2,120,200)
xorimg2 =cv.dilate(canny_img2,kernel,iterations = 1)


img_jian  = xorimg2-xorimg1
img_jian = cv.medianBlur(img_jian,3)
img_jian =cv.dilate(img_jian,kernel,iterations = 1)
img_jian = cv.dilate(img_jian,kernel,iterations = 2)
img_jian = cv.erode(img_jian,kernel,iterations = 2)
# img_jian =cv.morphologyEx(img_jian, cv.MORPH_CLOSE, kernel,3)
#

img_jian,contours,hier = cv.findContours(img_jian,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

for alist in contours:
    max_X = 0
    max_Y = 0
    min_X = img_res.shape[1]
    min_Y = img_res.shape[0]
    for blist in alist:
        for clist in blist:
            max_X = max(max_X, clist[0])
            max_Y = max(max_Y, clist[1])
            min_X = min(min_X, clist[0])
            min_Y = min(max_Y, clist[1])
    print(max_X, max_Y, min_X, min_Y)
    if max_X != min_X and max_Y != min_Y and (max_Y-min_Y)*(max_X-min_X)>250:
        cv.rectangle(img_res, (min_X, min_Y), (max_X, max_Y), (255, 255, 0), 2)

cv.drawContours(img_res,contours,-1,(255,0,255),2)

cv.imshow('img_res',img_res)
cv.imshow('xorimg2',xorimg2)
cv.imshow('img_jian',img_jian)
cv.waitKey()