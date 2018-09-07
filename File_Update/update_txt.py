import os
import numpy as np
import cv2 as cv
paths='E:\Car_Person2'
for file in os.listdir(paths):
    if file[-4:]=='.txt': ## and file[:-4]>'00500' and file[:-4]>'E:/BOT_Person/trainb/000000480594.jpg' and file[:-4]>'000000480591'  and file[:-4]>'84000'
        new_box=''
        new_txt =open(paths+'/'+file)
        old_data = new_txt.read()
        new_txt.close()
        img=cv.imread(paths+'/'+file[:-4]+'.jpg')
        output_txt = ''
        for bbox in old_data.split('\n'):
            if len(bbox)>3:
                output_txt +='1'+bbox[1:]+'\n'
        print(output_txt)
        write_txt = open(paths +'/'+ file[:-4] + '.txt', 'w')
        write_txt.write(output_txt)
        write_txt.close()

