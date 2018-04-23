import os
import numpy as np
import tensorflow as tf




def read_text():
    face_path = []
    noface_path = []
    face_lables = []
    noface_lables = []
    for file in os.listdir('D:/bot/face_regress/neg'):
        noface_path.append('D:/bot/face_regress/neg'+'/'+file)
        noface_lables.append(0)
    for file in os.listdir(r'D:/bot/face_regress/image_face_into'):
        face_path.append(r'D:/bot/face_regress/image_face_into'+'/'+file)
        face_lables.append(1)
    image_list=np.hstack((face_path,noface_path))
    lable_list = np.hstack((face_lables, noface_lables))
    temp=np.array([image_list,lable_list])
    temp=temp.transpose()
    np.random.shuffle(temp)
    all_image=list(temp[:,0])
    all_lable=list(temp[:,1])
    return all_image, all_lable


read_text()