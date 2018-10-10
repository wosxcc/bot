


import numpy as np
import os



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
face_img = np.array(face_img, dtype='str')
face_lab = np.array(face_lab, dtype='int')

print('face_img',face_img.shape,face_img[-20:])

print('face_lab',face_lab.shape,face_lab[-20:])

# img,lab = get_data_img()
#
# print('img',img.shape,img[-20:])
#
# print('lab',lab.shape,lab[-20:])
#
# print(lab[-1]+1)
