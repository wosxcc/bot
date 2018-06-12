# -*- coding: UTF-8 -*-
from aip import AipFace
import cv2 as cv
import json
import matplotlib.pyplot as plt
""" 你的 APPID AK SK """
APP_ID = '10454819'
API_KEY = 'q2bA1wTuhPtoGRoYp0ROXUwQ'
SECRET_KEY = '0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo'

aipFace = AipFace(APP_ID, API_KEY, SECRET_KEY)

print(aipFace)

imageType = "BASE64"
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()
# 读取图片
filePath = '../face_into/face68/image_test/sface_2877.jpg'

img=cv.imread(filePath)
cv.imshow('img',img)
cv.waitKey(0)


        # 定义参数变量


options = {
    'max_face_num': 1,
    'face_fields': 'landmark72',
}
# 调用人脸属性检测接口
image  = get_file_content(filePath)

print(image)
# result = aipFace.detect(get_file_content(filePath), options)
result = aipFace.detect(image, options)

aface = result.encode('utf-8')


# print(result)
# print(type(result))

# 解析位置信息
location=result['result'][0]['location']
left_top=(location['left'],location['top'])
right_bottom=(left_top[0]+location['width'],left_top[1]+location['height'])


cv.rectangle(img,left_top,right_bottom,(0,0,255),2)

cv.imshow('img',img)
cv.waitKey(0)
# plt.imshow(img,"gray")
# plt.show()

# image= "../face_into/face68/image_test/sface_2877.jpg"
# imageType = "BASE64"
# client = AipFace(APP_ID, API_KEY, SECRET_KEY)
# """ 调用人脸检测 """
# cccc=client.detect(image, imageType);
# print(cccc)
# """ 如果有可选参数 """
# options = {}
# options["face_field"] = "age"
# options["max_face_num"] = 2
# options["face_type"] = "LIVE"
#
# """ 带参数调用人脸检测 """
# xxxx=client.detect(image, imageType, options)
#
# print(xxxx)