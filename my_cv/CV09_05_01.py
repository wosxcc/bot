import cv2 as cv
import numpy as np
import random
# img1 = cv.imread('E:/Desk_why/1.jpg')
from math import *

#             打印机         电话            计算器            文件架       水杯
bianchang = {'0':[160,260],'1':[120,160],'2':[80,100],'3':[150,250],'4':[25,80]}




img = cv.imread('E:/Desk_Set/hhhh/0/1.jpg')

cv.imshow('img',img)
cv.waitKey()

height,weight,ch =img.shape



degree=random.randint(-45,45)
print(degree)
#旋转后的尺寸
heightNew=int(weight*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
widthNew=int(height*fabs(sin(radians(degree)))+weight*fabs(cos(radians(degree))))

matRotation=cv.getRotationMatrix2D((weight/2,height/2),degree,1)

matRotation[0,2] +=(widthNew-weight)/2  #重点在这步，目前不懂为什么加这步
matRotation[1,2] +=(heightNew-height)/2  #重点在这步

imgRotation=cv.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))

cv.imshow('imgRotation',imgRotation)
pts1 = np.float32([[int(height/10),int(height/10)],[int(weight/2),int(height/2)],[height,weight]])
pts2 = np.float32([[int(height/20),0],[int(weight/2),int(height/2)],[height/1.1,int(weight)]])
m= cv.getAffineTransform(pts1,pts2)
dst=cv.warpAffine(img,m,(height,weight))
cv.imshow('dst',dst)
cv.imshow('img',img)
cv.waitKey()