import os
import numpy as np
import cv2 as cv

import random

def image_face_id():
    img_path = 'E:/Face_Data/lfwcrop_color/faces'


    faceids  ={}
    for file in os.listdir(img_path):

        if file[:-9] not in faceids:
            print(file[:-9])
            faceids[file[:-9]]=[]
        faceids[file[:-9]].append(file)
    count_name = 0
    img_data =[]
    for face_name in faceids:
        if len(faceids[face_name])<2:
            continue
        else:
            print(len(faceids[face_name])-1)
            for i in range(len(faceids[face_name])-1):
                while(1):
                    noface = random.sample(faceids.keys(), 1)[0]
                    if noface!=face_name:
                        break
                print('结构列',[img_path+'/'+faceids[face_name][0],img_path+'/'+faceids[face_name][i+1],img_path+'/'+faceids[noface][0]])
                # img =cv.imread(img_path+'/'+faceids[face_name][i+1])
                # cv.imshow('img',img)
                # cv.waitKey()
                img_data.append([img_path+'/'+faceids[face_name][0],img_path+'/'+faceids[face_name][i+1],img_path+'/'+faceids[noface][0]])
            count_name+=1

    return img_data


# image_lines =image_face_id()
# print('哈哈哈',len(image_lines))
# label_lines =np.zeros((len(image_lines),128))
# print(label_lines.shape)
# print(image_lines[1])