import h5py
import cv2 as cv
import numpy as np
import random

img_bian=12

file_yuan=open('train_yes.txt')
get=file_yuan.read()
result=get.split('\n')
other_result=get.splitlines()
img_data=np.zeros((len(other_result),3,img_bian,img_bian))
lab_data=np.zeros((len(other_result),1))
freq_data=np.zeros((len(other_result),4))
f = h5py.File('HDF612_train.h5','w')

random.shuffle(other_result)
for i in  range(len(other_result)):
    data_list = other_result[i].split('\t')
    img=cv.imread('./'+data_list[0])
    if img.shape[0]!=img_bian:
        img=cv.resize(img,None,fx=float(img_bian/img.shape[1]),fy=float(img_bian/img.shape[1]), interpolation=cv.INTER_CUBIC)
    imgxx=img.astype(np.float32)/256.0

    lables=float(data_list[1])
    # print(int(data_list[1]))
    # lables[int(data_list[1])]=1
    freq=[float(data_list[2]), float(data_list[3]), float(data_list[4]), float(data_list[5])]
    imgxx = imgxx[:, :, [2, 1, 0]]
    imgxx = imgxx.transpose((2, 0, 1))
    print('第{}张图片'.format(i),imgxx.shape)
    img_data[i]=imgxx
    lab_data[i]=lables
    freq_data[i]=freq
    # cv.imshow('img',img)
    # cv.waitKey()

f['data']=img_data #这个地方的名字决定了你使用caffe时候top的名称，名称没对应上是错误的。
f['label']=lab_data #同上面
f['freq']=freq_data
print(img_data.shape)
print(lab_data.shape)
print(freq_data.shape)
print(img_data.shape,img_data[img_data.shape[0]-11:img_data.shape[0]-1])
print(lab_data.shape,lab_data[lab_data.shape[0]-11:lab_data.shape[0]-1])
h5txt_file = open('h5txt_path.txt', 'w')
h5txt_file.write('HDF512_train.h5')
h5txt_file.close()
f.close()


