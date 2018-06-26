# coding:gb2312
import cv2 as cv
from PIL import ImageGrab
from win32api import GetSystemMetrics
import numpy as np

bbox = (0, 0,  GetSystemMetrics (0), GetSystemMetrics (1))   #####(sx,sy,ex,ey)
im = ImageGrab.grab(bbox)   ###获取屏幕图像
# im.save('as.jpg')
img = np.array(im)       ###转化为array数组
img =cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow('img',img)
cv.waitKey()
# 参数 保存截图文件的路径
