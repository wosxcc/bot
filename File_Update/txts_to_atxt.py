import os
import cv2 as cv




path_files = 'E:/dectect/dectect/face68'

atxt_open = open('trains.txt','w')
count_img =0
for file in os.listdir(path_files):
    if (file[-4:]=='.txt'):
        file_open=open(path_files+ '/'+ file,'r')
        file_read = file_open.read()
        count_str = str(count_img)
        for i in range(5-len(count_str)):
            count_str='0'+count_str
        file_open.close()
        img_name='E:/face72/trains/'+ count_str + '.jpg'
        atxt_open.write( img_name + ' ' + file_read + '\n')
        img =cv.imread(path_files+ '/'+ file[:-4] + '.jpg')
        img= cv.resize(img,(96,96),cv.INTER_CUBIC)
        cv.imwrite( img_name,img)
        count_img+=1
        # cv.imshow('img',img)
        # cv.waitKey()
print(count_img)
atxt_open.close()
