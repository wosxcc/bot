import cv2 as cv
import numpy as np


img1 = cv.imread('./Desk/184.jpg')

img1 =cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img1 =cv.resize(img1,(600,600),cv.INTER_CUBIC)
img2 = cv.imread('./Desk/170.jpg')
img2 =cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
img2 =cv.resize(img2,(600,600),cv.INTER_CUBIC)


print(img1.shape)
img_H ,img_W =img1.shape


imgc = np.zeros(img1.shape,dtype='uint8')
imgd = np.zeros(img1.shape,dtype='uint8')
imge = np.zeros(img1.shape,dtype='uint8')

for i in range(img_W):
    for j in range(img_H):
        imgd[j][i] = img1[j][i]-img2[j][i]
        imge[j][i] = (img1[j][i]//25)-(img2[j][i]//25)
        if abs(img2[j][i]-img1[j][i])>50:
            imgc[j][i] = 255
cv.imshow('imgc',imgc)
cv.imshow('imgd',imgd)

cv.imshow('imge',imge)
cv.waitKey()