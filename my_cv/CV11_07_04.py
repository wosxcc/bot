import cv2 as cv
import numpy as np

img1 = cv.imread('./Desk/184.jpg')
imgss =cv.resize(img1,(600,600),cv.INTER_CUBIC)
img1 =cv.cvtColor(imgss,cv.COLOR_BGR2GRAY)

img2 = cv.imread('./Desk/170.jpg')

imgssb =cv.resize(img2,(600,600),cv.INTER_CUBIC)
img2 =cv.cvtColor(imgssb,cv.COLOR_BGR2GRAY)


img1 =cv.blur(img1, (3, 3))
canny_img1 = cv.Canny(img1,100,200)


img2=cv.blur(img2, (3, 3))
canny_img2 = cv.Canny(img2,100,200)





xorimg =cv.bitwise_xor(canny_img2,canny_img1)
# kernel = np.ones((2,2),np.uint8)
# xorimg =cv.morphologyEx(xorimg,cv.MORPH_OPEN,kernel)
image,contours,hier = cv.findContours(xorimg,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
for alist in contours:
    max_X = 0
    max_Y = 0
    min_X = image.shape[1]
    min_Y = image.shape[0]
    for blist in alist:
        for clist in blist:
            max_X = max(max_X,clist[0])
            max_Y = max(max_Y,clist[1])
            min_X = min(min_X,clist[0])
            min_Y = min(max_Y,clist[1])

    print(max_X,max_Y,min_X,min_Y)
    if max_X !=min_X and max_Y!=min_Y:
        cv.rectangle(imgssb,(min_X,min_Y),(max_X,max_Y),(255,255,0),2)

cv.drawContours(imgssb,contours,-1,(255,0,255),4)
cv.imshow("imgsscc",xorimg)
kernel = np.ones((3,3),np.uint8)
xorimg =cv.erode(xorimg,kernel,iterations = 1)
# xorimg =cv.morphologyEx(xorimg,cv.MORPH_CLOSE,kernel,3)

# cv.imshow("canny_img1",canny_img1)
cv.imshow("imgss",imgss)
cv.imshow("imgssb",imgssb)
# cv.imshow("canny_img2",canny_img2)
cv.imshow('xorimg',xorimg)
cv.waitKey(0)



