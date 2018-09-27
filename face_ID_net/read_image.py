import os
import numpy as np
import cv2 as cv

import random

def image_face_id():
    img_path = 'E:/Face_Data/lfwcrop_color/faces'
    faceids  ={}
    for file in os.listdir(img_path):
        if file[:-9] not in faceids:
            # print(file[:-9])
            faceids[file[:-9]]=[]
        faceids[file[:-9]].append(file)
    count_name = 0
    img_data =[]
    for face_name in faceids:
        if len(faceids[face_name])<2:
            continue
        else:
            # print(len(faceids[face_name])-1)
            for i in range(len(faceids[face_name])-1):
                while(1):
                    noface = random.sample(faceids.keys(), 1)[0]
                    if noface!=face_name:
                        break
                # print('结构列',[img_path+'/'+faceids[face_name][0],img_path+'/'+faceids[face_name][i+1],img_path+'/'+faceids[noface][0]])
                # img =cv.imread(img_path+'/'+faceids[face_name][i+1])
                # cv.imshow('img',img)
                # cv.waitKey()
                img_data.append([img_path+'/'+faceids[face_name][0],img_path+'/'+faceids[face_name][i+1],img_path+'/'+faceids[noface][0]])
            count_name+=1

    return img_data


def read_image():
    imgread_date = []

    image_lines =image_face_id()
    for agroup in image_lines:
        group_img =[]
        for imgc in agroup:
            img = cv.imread(imgc)
            img =cv.resize(img ,(64,64),interpolation=cv.INTER_CUBIC)
            img =np.array(img,dtype='float32')/256.0
            group_img.append(img)

        imgread_date.append(group_img)
    return imgread_date

