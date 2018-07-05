from matplotlib import pyplot as plt
import cv2  as cv# used for resize. if you dont have it, use anything else
import numpy as np
from model import Deeplabv3
import datetime
deeplab_model = Deeplabv3()
img = cv.imread("./imgs/psb1.jpg")

cv.imshow('img',img)
# cv.waitKey()
w, h, _ = img.shape
ratio = 512. / np.max([w,h])

stime =datetime.datetime.now()
resized = cv.resize(img,(int(ratio*h),int(ratio*w)))
resized = resized / 127.5 - 1.
pad_x = int(512 - resized.shape[0])
resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
res = deeplab_model.predict(np.expand_dims(resized2,0))

print('耗时：',datetime.datetime.now()-stime)

labels = np.argmax(res.squeeze(),-1)

cv.imshow('image',labels[:-pad_x])
plt.imshow(labels[:-pad_x])
plt.show()