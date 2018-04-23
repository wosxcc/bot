import csv
import os
import cv2 as cv
import numpy as np
file_txt = open('train_yes.txt')
get = file_txt.read()
result = get.split('\n')
other_result = get.splitlines()
xcount=0
for i in range(len(other_result)):
    if other_result[i][:5]=='image':
        xcount+=1
        print('第{0}张图片'.format(xcount))
        new_name=str(xcount)
        xxresual=other_result[i].split('\t')
        # print(xxresual)
        new_data='0'
        img_X = (float(xxresual[2]) + float(xxresual[4])) / 2.0
        img_Y = (float(xxresual[3]) + float(xxresual[5])) / 2.0
        img_W = (float(xxresual[4]) - float(xxresual[2])) / 2.0
        img_H = (float(xxresual[5]) - float(xxresual[3])) / 2.0
        img=cv.imread(xxresual[0])
        cv.imwrite('./img_train/'+new_name+'.jpg',img)
        new_data += ' ' + str(img_X) + ' ' + str(img_Y) + ' ' + str(img_W) + ' ' + str(img_H)
        # print(new_data)
        out_file = open('./img_train/'+new_name+ '.txt', 'w')
        out_file.write( new_data+ '\n')
        out_file.close()