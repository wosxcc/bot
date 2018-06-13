import requests
import base64
from aip import AipFace
import cv2 as cv
import urllib3
import json
import os
# url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general?access_token=24.f9ba9c5241b67688bb4adbed8bc91dec.2592000.1485570332.282335-8574074'
ak = 'ak'
sk = 'sk'
host = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=q2bA1wTuhPtoGRoYp0ROXUwQ&client_secret=0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo'
# host = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=LUGBatgyRGoerR9FZbV4SQYk&client_secret=fB2MNz1c2UHLTximFlC4laXPg7CVfyjV'


ret = requests.get(host).json()

headers = {
    'Content-Type': 'application/json'
}

file_path= 'E:/face into'
for file in os.listdir(file_path):
    img_path=file_path+'/'+file
# img_path = '../face_into/face68/image_test/sface_4241f.jpg'
    with open(img_path, mode='rb') as f:
        img_base64 = bytes.decode(base64.b64encode(f.read()))


    # data=json.dumps({'image':img_base64,'image_type':'BASE64','face_type':'LIVE','max_face_num':'1','face_field': 'landmark'})
    # # data = {
    # #     'image': img_base64,
    # #     'image_type': 'BASE64',
    # #     'face_field': 'age'
    # #     # 'url': img_url
    # # }
    # url = f"https://aip.baidubce.com/rest/2.0/face/v3/detect?access_token={ret['access_token']}"
    # print(data)
    # print(url)
    #
    # ans = requests.post(url, data=data, headers=headers)
    # print(ans.text)
    # # print(ans['result'][0]['location'])
    # for xxx in ans:
    #     print('我去',xxx)

    #
    APP_ID = '10454819'
    API_KEY = 'q2bA1wTuhPtoGRoYp0ROXUwQ'
    SECRET_KEY = '0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo'

    client = AipFace(APP_ID, API_KEY, SECRET_KEY)
    options = {}
    options['face_field'] = 'age,landmark,expression'
    options['max_face_num'] = 1
    options['face_type'] = 'LIVE'
    xcc=client.detect(img_base64, 'BASE64', options=options)
    print(xcc)
    if xcc['result'] !=None:
        location=xcc['result']['face_list'][0]['location']
        left_top=(int(location['left'])*10,int(location['top'])*10)


        right_bottom=(int(left_top[0]+location['width'])*10,int(left_top[1]+location['height'])*10)
        img=cv.imread(img_path)

        img=cv.resize(img, None, fx=10, fy=10, interpolation=cv.INTER_CUBIC)

        cv.rectangle(img,left_top,right_bottom,(0,0,255),2)



        face72=xcc['result']['face_list'][0]['landmark72']
        for face_init in face72:
            cv.circle(img,(int(face_init['x'])*10,int(face_init['y'])*10),2,(0,255,0),-1)

        expression=xcc['result']['face_list'][0]['expression']
        # print(expression)
        # biaoq='wxiao'
        # if expression['type']=='none':
        #     biaoq='不笑'
        # elif expression['type'] == 'laugh':
        #     biaoq='大笑'

        biaoq=expression['type']+':'+str(expression['probability'])
        img = cv.putText(img,biaoq,(20,40),4, cv.FONT_HERSHEY_PLAIN, (255, 0, 0))
        cv.imshow('img',img)
        cv.waitKey(0)
