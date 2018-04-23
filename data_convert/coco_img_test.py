import cv2 as cv
import  numpy as np


img =cv.imread('E:/xcc_download/train2017/000000581674.jpg')
cv.imshow('img',img)

cv.waitKey()
print(img.shape)
# box=  [0.0, 176.0, 17.0, 22.0]
box=  [523.0, 345.0, 21.0, 9.0]
box= [int(i) for i in box]
img=cv.rectangle(img,(box[0], box[1]),(box[0]+box[2], box[1]+box[3]),(0,255,0),2)

img=cv.ellipse(img, (box[0], box[1]), (box[2], box[3]), 0, 0, 360, (0, 255, 255), 2)

cv.imshow('img',img)

cv.waitKey()
cv.destroyAllWindows()

##'size': [478, 640]}, 'area': 1734.0, 'iscrowd': 0, 'image_id': 101084, 'bbox': [150.0, 216.0, 54.0, 58.0], 'category_id': 130, 'id': 10040915}, {'40917