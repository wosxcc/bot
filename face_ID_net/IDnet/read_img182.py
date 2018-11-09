


import numpy as np
import os

import random

#
#
# def get_data_img():
#     img_path = 'E:/about_Face/facenet-master/data/casia_maxpy_mtcnnpy_182'
#     face_img = []
#     face_lab = []
#     for spath in os.listdir(img_path):
#         for file in os.listdir(img_path+'/'+spath):
#             if file[:-9] not in faceids:
#
#     face_img = np.array(face_img,dtype='str')
#     face_lab = np.array(face_lab,dtype='int')
#     return face_img,face_lab



img_path = 'E:/about_Face/facenet-master/data/casia_maxpy_mtcnnpy_182'
face_img = []
face_lab = []
count = 0
for spath in os.listdir(img_path):
    for file in os.listdir(img_path + '/' + spath):
        face_img.append(img_path + '/' + spath+'/'+file)
        face_lab.append(count)
    count+=1
# face_img = np.array(face_img, dtype='str')
# face_lab = np.array(face_lab, dtype='int')
img_train = []
lab_train = []
for i in range(1000):
    while True:
        index_epoch = []
        for i in range(5):
            xxx = random.randint(0, len(face_lab) - 6)
            index_epoch.append(xxx)
            index_epoch.append(xxx + 1)
            index_epoch.append(xxx + 2)
            index_epoch.append(xxx + 3)
            index_epoch.append(xxx + 4)
            index_epoch.append(xxx + 5)

        label_epoch = np.array(face_lab)[index_epoch]

        label_name ={}

        for alabel in label_epoch:
            if str(alabel) not in label_name:
                label_name[str(alabel)] = 1
            else:
                label_name[str(alabel)] += 1
        max_name =max(label_name, key=label_name.get)
        if label_name[max_name]>1:
            break

    image_epoch = np.array(face_img)[index_epoch]
    img_train.extend(image_epoch)
    lab_train.extend(label_epoch)



print('img_train',len(img_train),img_train)
print('lab_train',len(lab_train),lab_train)


# print('face_img',face_img.shape,face_img[-20:])
# print('face_lab',face_lab.shape,face_lab[-20:])

# img,lab = get_data_img()
#
# print('img',img.shape,img[-20:])
#
# print('lab',lab.shape,lab[-20:])
#
# print(lab[-1]+1)
