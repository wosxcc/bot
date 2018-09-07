import cv2 as cv
import numpy as np
import os
import random
import shutil

file_path = 'Danger_car' # './train_person'

out_path = './Danger/' # './person_libary/'
# for file in os.listdir(file_path):
count =24
inpot_path = './img_car'
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
    random_biasx = random.randint(-240, 240)
    random_biasy = random.randint(-240, 240)
    # print(random_biasx,'random_biasx',random_biasy,'random_biasy')
    for line in open_read.split('\n'):
        if len(line)>5:
            bbox = [float(i) for  i in  line.split(' ')]
            pers_x1 = int(bbox[1] * 480)- int(bbox[3] * 480 / 2)
            pers_y1 = int(bbox[2] * 480 )- int(bbox[4] * 480 / 2)

            pers_x2 = int(bbox[1] * 480 ) + int(bbox[3] * 480 / 2)
            pers_y2 = int(bbox[2] * 480 ) + int(bbox[4] * 480 / 2)
            imgc =imgb[pers_y1:pers_y2,pers_x1:pers_x2]
            bei = random.randint(8, 30) / 10.0
            imgc =cv.resize(imgc,None, fx=bei, fy=bei, interpolation=cv.INTER_CUBIC)
            img_dark = random.randint(6,10)/10.0

            # img_dark = random.randint(-10, 20)
            output_txt=''

            for xi in range(0, imgc.shape[1]):
                for xj in range(0, imgc.shape[0]):
                    # set the pixel value decrease to 20%
                    imgc[xj, xi, 0] = int(imgc[xj, xi, 0] * img_dark)
                    imgc[xj, xi, 1] = int(imgc[xj, xi, 1] * img_dark)
                    imgc[xj, xi, 2] = int(imgc[xj, xi, 2] * img_dark)
                    #
                    # imgc[xj, xi, 0] = int(imgc[xj, xi, 0] + img_dark)
                    # imgc[xj, xi, 1] = int(imgc[xj, xi, 1] + img_dark)
                    # imgc[xj, xi, 2] = int(imgc[xj, xi, 2] + img_dark)
            # print(img_dark)
            # print(type(img_dark))
            # print(type(img_dark.dtype('u1')))
            # imgc[:,:,:]*=img_dark

            now_y1 = min(max(pers_y1 + random_biasy,0),480-imgc.shape[0])
            now_y2 = now_y1+imgc.shape[0]
            now_x1 = min(max(pers_x1 + random_biasx,0),480-imgc.shape[1])
            now_x2 = now_x1+imgc.shape[1]
            # print(imgc.shape,pers_y2-pers_y1,pers_x2-pers_x1)
            out_x = float(now_x1+imgc.shape[1]/2.0)/480.0
            out_y = float(now_y1+imgc.shape[0]/2.0)/480.0
            out_w = float(imgc.shape[1])/480.0
            out_h = float(imgc.shape[0])/480.0

            output_txt +='0 '+str(out_x)+ ' '+ str(out_y)+ ' '+str(out_w)+ ' '+str(out_h)+ '\n'

            imga[now_y1:now_y2,now_x1:now_x2] =imgc

    # shutil.copyfile(file_path+'/'+file[:-4]+'.txt', out_path+str(count)+'.txt')
    write_txt = open(out_path+str(count)+'.txt', 'w')
    write_txt.write(output_txt)
    write_txt.close()
    cv.imwrite( out_path+str(count)+'.jpg',imga)
    # cv.imshow('imgb',imgb)
    # cv.imshow('imga', imga)
    # cv.waitKey()
    count+=1
    print('当前第%d张图片'%(count))
    # cv.waitKey()
