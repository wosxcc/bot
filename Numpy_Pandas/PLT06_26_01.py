# coding:gb2312
import cv2 as cv
from PIL import ImageGrab
from win32api import GetSystemMetrics
import numpy as np

bbox = (0, 0,  GetSystemMetrics (0), GetSystemMetrics (1))   #####(sx,sy,ex,ey)
im = ImageGrab.grab(bbox)   ###��ȡ��Ļͼ��
# im.save('as.jpg')
img = np.array(im)       ###ת��Ϊarray����
img =cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow('img',img)
cv.waitKey()
# ���� �����ͼ�ļ���·��
