import requests
import base64
from aip import AipFace
import cv2 as cv
import urllib3
import json
import os
import datetime

file_path= 'E:/dectect/dectect/face68'

file_path= 'E:/dectect/dectect/face68'
for file in os.listdir(file_path):
    img_path=file_path+'/'+file
# img_path = '../face_into/face68/image_test/sface_4241f.jpg'
    with open(img_path, mode='rb') as f:
        img_base64 = bytes.decode(base64.b64encode(f.read()))
    APP_ID = '10454819'
    API_KEY = 'q2bA1wTuhPtoGRoYp0ROXUwQ'
    SECRET_KEY = '0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo'

    client = AipFace(APP_ID, API_KEY, SECRET_KEY)
    options = {}
    options['face_field'] = 'age,landmark,expression'
    options['max_face_num'] = 1
    options['face_type'] = 'LIVE'
    stime=datetime.datetime.now()
    xcc=client.detect(img_base64, 'BASE64', options=options)
    print('耗时：',datetime.datetime.now()-stime)
    print(xcc)
    if 'result' in xcc:
        if xcc['result'] !=None :
            img = cv.imread(img_path)
            print(xcc)
            location=xcc['result']['face_list'][0]['location']
            left_top=(int(location['left'])*10,int(location['top'])*10)
            right_bottom=(int(left_top[0]+location['width'])*10,int(left_top[1]+location['height'])*10)
            expression = xcc['result']['face_list'][0]['expression']
            face72=xcc['result']['face_list'][0]['landmark72']
            print(face72)
            for face_init in face72:
                img=cv.circle(img,(int(face_init['x']),int(face_init['y'])), 2,(0,255,0),-1)
            biaoq=expression['type']+':'+str(expression['probability'])
            img = cv.putText(img,biaoq,(5,30),4, cv.FONT_HERSHEY_PLAIN, (255, 0, 0))
            cv.imshow('img',img)
            cv.waitKey( )
