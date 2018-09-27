import os
import cv2 as cv
import numpy as np


def get_img_data():
    img_data ={}
    path= 'E:/Face_ID/'
    for i in range(10):
        i+=1
        for img_name in os.listdir(path+str(i)+'face'):
            # img = cv.imread(path+str(i)+'face/'+img_name)
            if img_name[:-9] not in img_data:
                img_data[img_name[:-9]]=[]
            img_data[img_name[:-9]].append(path+str(i)+'face/'+img_name)
    X_data = []
    for name  in img_data:
        group_img =[]
        for path_img in img_data[name]:
            img = cv.imread(path_img)
            img =cv.resize(img,(96,96),interpolation=cv.INTER_CUBIC)
            imgd = np.array(img,dtype='float32')/256
            group_img.append(imgd)
            # print(imgd.shape)
            # print(len(group_img))
            # cv.imshow('img',img)
            # cv.waitKey()
        X_data.append(group_img)
    # print(len(X_data))
    return  X_data

data = get_img_data()
print('data',data[:10])