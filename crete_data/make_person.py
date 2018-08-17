import cv2 as cv
import numpy as np
import os
import random
import shutil

file_path= './train_person'

out_path ='./person_libary/'
# for file in os.listdir(file_path):
count =99149
inpot_path = 'E:/NO_person'
for img_name  in os.listdir(inpot_path):
    imga = cv.imread(inpot_path+'/'+img_name)
    imga =cv.resize(imga,(480,480),interpolation=cv.INTER_CUBIC)
    while(1):
        file=random.sample(os.listdir(file_path),1)[0]
        if file[-4:]== '.jpg':
            break
    imgb = cv.imread(file_path+'/'+file)
    open_txt = open(file_path+'/'+file[:-4]+'.txt')
    open_read =open_txt.read()
    for line in open_read.split('\n'):
        if len(line)>5:
            bbox = [float(i) for  i in  line.split(' ')]
            pers_x1 = max(int(bbox[1] * 480)- int(bbox[3] * 480 / 2)-10,0)
            pers_y1 = max(int(bbox[2] * 480)- int(bbox[4] * 480 / 2)-10,0)

            pers_x2 = min(int(bbox[1] * 480) + int(bbox[3] * 480 / 2)+10,480)
            pers_y2 = min(int(bbox[2] * 480) + int(bbox[4] * 480 / 2)+10,480)
            imgc =imgb[pers_y1:pers_y2,pers_x1:pers_x2]
            imga[pers_y1:pers_y2,pers_x1:pers_x2] =imgc

    shutil.copyfile(file_path+'/'+file[:-4]+'.txt', out_path+str(count)+'.txt')
    cv.imwrite( out_path+str(count)+'.jpg',imga)
    # cv.imshow('imgb',imgb)
    # cv.imshow('imga', imga)
    count+=1
    print('当前第%d张图片'%(count))
    # cv.waitKey()
