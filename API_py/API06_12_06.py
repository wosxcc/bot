from aip import AipFace
import base64
""" 你的 APPID AK SK """
APP_ID = '10454819'
API_KEY = 'q2bA1wTuhPtoGRoYp0ROXUwQ'
SECRET_KEY = '0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

imageType = "BASE64"
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image=get_file_content('../face_into/face68/image_test/sface_2877.jpg')

""" 调用人脸检测 """

print(imageType)
print(image)
print (client.detect(image, imageType))

""" 如果有可选参数 """
options = {}
options["face_field"] = "age"
options["max_face_num"] = 2
options["face_type"] = "LIVE"

""" 带参数调用人脸检测 """
client.detect(image, imageType, options)
print(client.detect(image, imageType, options))