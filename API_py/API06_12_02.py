# -*- coding:utf-8 -*-
from aip import AipFace
import cv2 as cv
import json
import matplotlib.pyplot as plt
""" 你的 APPID AK SK """
APP_ID = '10454819'
API_KEY = 'q2bA1wTuhPtoGRoYp0ROXUwQ'
SECRET_KEY = '0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)
imageType = 'BASE64'
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('../face_into/face68/image_test/sface_2877.jpg')

result=client.detect(image);
# 解析位置信息
location=result['result'][0]['location']
left_top=(location['left'],location['top'])
right_bottom=(left_top[0]+location['width'],left_top[1]+location['height'])
myimage='faceapitest.png'
img=cv2.imread(myimage)
cv2.rectangle(img,left_top,right_bottom,(0,0,255),2)
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()