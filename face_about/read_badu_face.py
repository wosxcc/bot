import os
import cv2 as cv
import numpy as np
txt_path ='E:/about_Face/faceID1'
img_path ='E:/about_Face/faceID'

for txt_flie in os.listdir(txt_path):
    img = cv.imread(img_path + '/' + txt_flie[:-6] + '.jpg')
    txt_open = open(txt_path+"/"+txt_flie)
    txt_read = txt_open.read()

    face_init =txt_read.split(' ')
    countt = 0
    for xx in face_init:
        countt+=1
    print("累计数量{}个参数".format(countt))


    print(face_init[7],face_init[6])
    for i in range(72):
        cv.circle(img,(int(float(face_init[6+2*i])),int(float(face_init[6+2*i+1]))),2,(0,0,255),-1)


    cv.putText(img, "ren zhong:"+face_init[-5], (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)
    cv.putText(img, "wei xiao: "+face_init[-4], (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)
    cv.putText(img, "xin bie:  "+face_init[-3], (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)
    cv.putText(img, "yan jing: "+face_init[-2], (0, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 1)
    cv.putText(img, "age:      "+face_init[-6], (0, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv.putText(img, "de fen:      " + face_init[150], (0, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    print("三个方向旋转角度",face_init[151]+ ' '+face_init[151]+' '+face_init[152])
    cv.putText(img, face_init[151]+ ' '+face_init[152]+' '+face_init[153], (0, 120), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv.putText(img, "jiaodu   :" + face_init[4], (0, 135), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv.rectangle(img,(int(float(face_init[0])),int(float(face_init[1]))),(int(float(face_init[0])+float(face_init[2])),int(float(face_init[1])+float(face_init[3]))),(255,0,0),2)

    cv.imshow('img',img)
    cv.waitKey()
    cv.destroyAllWindows()






