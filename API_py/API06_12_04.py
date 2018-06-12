# -*- coding: UTF-8 -*-

from aip import AipFace
import cv2
import base64



APP_ID = '10454819'
API_KEY = 'q2bA1wTuhPtoGRoYp0ROXUwQ'
SECRET_KEY = '0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo'

# 初始化AipFace对象
aipFace = AipFace(APP_ID, API_KEY, SECRET_KEY)

# 读取图片
filePath = "../face_into/face68/image_test/sface_2877.jpg"


def get_file_content(filePath):

    f = open(r'%s' % filePath, 'rb')
    pic1 = base64.b64encode(f.read())
    f.close()
    params = {'image': str(pic1,'UTF-8')}
    return params

    # with open(filePath, 'rb') as fp:
    #     return fp.read()

        # 定义参数变量


# options = {
#     'max_face_num': 1, # 图像数量
#     'face_fields': 'age,beauty,expression,faceshape',
# }

options = {}
options['face_field'] = 'age'
options['max_face_num'] = '1'
options['face_type'] = 'LIVE'
# 调用人脸属性检测接口
result = aipFace.detect(get_file_content(filePath), options)
print(result)
# print(result)
# print(type(result))

# 解析位置信息
location=result['result'][0]['location']
left_top=(location['left'],location['top'])
right_bottom=(left_top[0]+location['width'],left_top[1]+location['height'])

img=cv2.imread(filePath)
cv2.rectangle(img,left_top,right_bottom,(0,0,255),2)

cv2.imshow('img',img)
cv2.waitKey(0)