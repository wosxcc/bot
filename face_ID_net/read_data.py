import os
import cv2 as cv


img_path = 'E:/Face_Data/lfw_funneled/'

txt_path  = 'E:\Face_Data\lfw_funneled/pairs_01.txt'


txt_open = open(txt_path)
txt_read = txt_open.read()

count_group=0
for img_group in txt_read.split('\n\n'):

    img_xb =0
    for img_name in img_group.split('\n'):
        img = cv.imread(img_path+img_name)
        img = cv.resize(img,(250,250),interpolation=cv.INTER_CUBIC)
        imgc =img[:,20:230,:]
        cv.imshow(str(img_xb),imgc)
        img_xb+=1
    cv.waitKey()

    print(count_group)
    print(img_group)
    count_group+=1