from aip import AipFace
import base64
import cv2 as cv
""" 你的 APPID AK SK """
APP_ID = '10454819'
API_KEY = 'q2bA1wTuhPtoGRoYp0ROXUwQ'
SECRET_KEY = '0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

imageType = 'BASE64'
def get_file_content(filePath):
    f = open(r'%s' % filePath, 'rb')
    pic1 = base64.b64encode(f.read())
    f.close()
    params = {'images': str(pic1, 'UTF-8')}
    return params

print(123456)
image='../face_into/face68/image_test/sface_3140.jpg'

img=cv.imread(image)

# imgsss=base64.b64encode(img)
# print(imgsss)
# cv.imshow('img',img)
# cv.waitKey()


""" 调用人脸检测 """
print(get_file_content(image))
print (client.detect(get_file_content(image), imageType))

""" 如果有可选参数 """
options = {}
options['face_field'] = 'landmark72'
options['max_face_num'] = '2'
options['face_type'] = 'LIVE'

""" 带参数调用人脸检测 """
client.detect(image, imageType, options)
print(client.detect(image, imageType, options))