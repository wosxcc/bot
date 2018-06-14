from aip import AipFace
import cv2
import base64
""" 你的 APPID AK SK """
APP_ID = '10454819'
API_KEY = 'q2bA1wTuhPtoGRoYp0ROXUwQ'
SECRET_KEY = '0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo'
image = 'D:/pproject/ppop/image/xcc.jpg'
imageType = "BASE64"
groupIdList = "kaifa"
""" 调用人脸搜索 """
client= AipFace(APP_ID, API_KEY, SECRET_KEY)
with open(image,mode='rb') as f:
    image_base64 = bytes.decode(base64.b64encode(f.read()))

    print(client.search(image_base64, imageType, groupIdList))

    # face_token人脸标志
    # user_list 匹配的用户信息列表
    # +group_id用户所属的group_id
    # +user_id 用户的user_id
    # +user_info 注册用户时携带的user_info
    # score 用户的匹配得分



    """ 如果有可选参数 """
    options = {}
    options["quality_control"] = "NORMAL"   #图片质量控制 NONE: 不进行控制 LOW:较低的质量要求 NORMAL: 一般的质量要求 HIGH: 较高的质量要求 默认 NONE
    options["liveness_control"] = "LOW"     #活体检测控制 NONE: 不进行控制 LOW:较低的活体要求(高通过率 低攻击拒绝率) NORMAL: 一般的活体要求(平衡的攻击拒绝率, 通过率) HIGH: 较高的活体要求(高攻击拒绝率 低通过率) 默认NONE
    options["user_id"] = "002"              #当需要对特定用户进行比对时，指定user_id进行比对。即人脸认证功能。
    options["max_user_num"] = 3             #查找后返回的用户数量。返回相似度最高的几个用户，默认为1，最多返回20个。

    """ 带参数调用人脸搜索 """
    print(client.search(image_base64, imageType, groupIdList, options))