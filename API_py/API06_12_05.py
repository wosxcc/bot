from aip import AipFace
import cv2
import matplotlib.pyplot as plt

# 定义常量
# APP_ID = '9851066'
# API_KEY = 'LUGBatgyRGoerR9FZbV4SQYk'
# SECRET_KEY = 'fB2MNz1c2UHLTximFlC4laXPg7CVfyjV'

APP_ID = '10454819'
API_KEY = 'q2bA1wTuhPtoGRoYp0ROXUwQ'
SECRET_KEY = '0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo'

# 初始化AipFace对象
client = AipFace(APP_ID, API_KEY, SECRET_KEY)

# 读取图片
filePath = "../face_into/face68/image_test/sface_2877.jpg"

imageType = "BASE64"
def get_file_content(filePath):
    import base64
    f = open(r'%s' % filePath, 'rb')
    pic1 = base64.b64encode(f.read())
    f.close()
    params = {"images": str(pic1, 'utf-8')}
    return params
image=get_file_content(filePath)




ccc=client.detect(image,imageType)
print(ccc)

""" 如果有可选参数 """
options = {}
options["face_field"] = "age"
options["max_face_num"] = 1
options["face_type"] = "LIVE"
print(options)
""" 带参数调用人脸检测 """
xx=client.detect(image,imageType, options)
print(xx)