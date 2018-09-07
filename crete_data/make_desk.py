import cv2 as cv
import numpy as np
import os
import random
from math import *
import shutil
file_path = 'E:/Desk_Set/hhhh/' # './train_person'     假数据文件位置
out_path = './ccccc/' # './person_libary/'     输出文件目录
count =50
inpot_path = 'E:/Desk_why'                      # 原数据目录


bianchang = {'0':[160,260],'1':[120,160],'2':[80,100],'3':[150,250],'4':[50,160],'5':[50,160]}



for img_name  in os.listdir(inpot_path):
    imga = cv.imread(inpot_path+'/'+img_name)
    imga =cv.resize(imga,(800,800),interpolation=cv.INTER_CUBIC)
    xxcount = random.randint(2,5)
    print('在图片：',img_name,'中加入',xxcount,'个假数据')
    for i in range(xxcount):

        file_path_two =random.randint(0,5)
        file=random.sample(os.listdir(file_path+str(file_path_two)),1)[0]
        imgb = cv.imread(file_path+str(file_path_two)+'/'+file)

        degree = random.randint(-45, 45)
        # print(degree)
        # 旋转后的尺寸
        height, weight, ch = imgb.shape
        heightNew = int(weight * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + weight * fabs(cos(radians(degree))))
        matRotation = cv.getRotationMatrix2D((weight / 2, height / 2), degree, 1)
        matRotation[0, 2] += (widthNew - weight) / 2  # 重点在这步，目前不懂为什么加这步
        matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步
        imgb = cv.warpAffine(imgb, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

        random_biasx = random.randint(50, 750)
        random_biasy = random.randint(50, 750)
        print(random.randint(bianchang[str(file_path_two)][0], bianchang[str(file_path_two)][1]))
        bei = float(random.randint(bianchang[str(file_path_two)][0], bianchang[str(file_path_two)][1])) /float(max(imgb.shape[0],imgb.shape[1]))
        imgc =cv.resize(imgb,None, fx=bei, fy=bei, interpolation=cv.INTER_CUBIC)
        img_dark = random.randint(6,10)/10.0
        output_txt=''
        for xi in range(0, imgc.shape[1]):
            for xj in range(0, imgc.shape[0]):
                # set the pixel value decrease to 20%
                imgc[xj, xi, 0] = int(imgc[xj, xi, 0] * img_dark)
                imgc[xj, xi, 1] = int(imgc[xj, xi, 1] * img_dark)
                imgc[xj, xi, 2] = int(imgc[xj, xi, 2] * img_dark)

        now_y1 = min(max(random_biasy,0),800-imgc.shape[0])
        now_y2 = now_y1+imgc.shape[0]
        now_x1 = min(max(random_biasx,0),800-imgc.shape[1])
        now_x2 = now_x1+imgc.shape[1]
        # print(imgc.shape,pers_y2-pers_y1,pers_x2-pers_x1)
        out_x = float(now_x1+imgc.shape[1]/2.0)/800.0
        out_y = float(now_y1+imgc.shape[0]/2.0)/800.0
        out_w = float(imgc.shape[1])/800.0
        out_h = float(imgc.shape[0])/800.0

        output_txt +='0 '+str(out_x)+ ' '+ str(out_y)+ ' '+str(out_w)+ ' '+str(out_h)+ '\n'

        imga[now_y1:now_y2,now_x1:now_x2] =imgc

        # shutil.copyfile(file_path+'/'+file[:-4]+'.txt', out_path+str(count)+'.txt')
        # write_txt = open(out_path+str(count)+'.txt', 'w')
        # write_txt.write(output_txt)
        # write_txt.close()
    cv.imwrite( out_path+str(count)+'.jpg',imga)
    count += 1
    # cv.imshow('imga', imga)
    # print('当前第%d张图片'%(count))
    # cv.waitKey()
