import cv2 as cv



img=cv.imread('./imgs/psb1.jpg')

cv.imshow('yimg',img)
print(img.shape)
img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.imshow('img_gray',img_gray)
print(img_gray.shape)
cv.waitKey()

