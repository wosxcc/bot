import h5py
import cv2 as cv
import numpy as np
import random

img_bian=128

test_count=641
file_yuan=open('trainc.txt')
get=file_yuan.read()
result=get.split('\n')
other_result=get.splitlines()
img_data=np.zeros((len(other_result)-test_count,3,img_bian,img_bian))
lab_data=np.zeros((len(other_result)-test_count,1))
freq_data=np.zeros((len(other_result)-test_count,28))
f = h5py.File('HDFxcc_train.h5','w')

random.shuffle(other_result)
for i in  range(len(other_result)-test_count):
    # i+=641
    data_list = other_result[i].split(' ')
    # print(data_list)
    if len(data_list) > 3:
        img=cv.imread(data_list[0])
        if img.shape[0]!=img_bian:
            img=cv.resize(img,None,fx=float(img_bian/img.shape[1]),fy=float(img_bian/img.shape[1]), interpolation=cv.INTER_CUBIC)
        imgxx=img.astype(np.float32)/256.0

        lables=min(float(data_list[1]),1)
        freq=[]

        # freq.append(float(data_list[1]))
        # freq.append(float(data_list[2]))
        for x in range(14):
            freq.append(float(data_list[117 + 2 + x * 2]))
        for x in range(14):
            freq.append(float(data_list[117 + 2 + x * 2 + 1]))


            # cv.circle(img,(int(float(data_list[117 + 2 + x * 2])*128),int(float(data_list[117 + 2 + x * 2+1])*128)),2,(0, 255, 255), -1)



        # freq=[float(data_list[2]), float(data_list[3]), float(data_list[4]), float(data_list[5])]
        imgxx = imgxx[:, :, [2, 1, 0]]
        # print(imgxx.shape)
        imgxx = imgxx.transpose((2, 0, 1))
        # print(imgxx.shape)
        print('第{}张图片'.format(i),imgxx.shape)
        img_data[i]=imgxx
        lab_data[i]=lables


        # print(freq)
        # print(len(freq))

        freq_data[i]=freq

        # print(lables)
        # print(len(freq))
        # cv.imshow('img',img)
        # cv.waitKey()

f['data']=img_data #这个地方的名字决定了你使用caffe时候top的名称，名称没对应上是错误的。
f['label']=lab_data #同上面
f['freq']=freq_data
print(f)
print(img_data.shape)
# print(lab_data.shape)
print(freq_data.shape)
print(img_data.shape,img_data[img_data.shape[0]-11:img_data.shape[0]-1])
# print(lab_data.shape,lab_data[lab_data.shape[0]-11:lab_data.shape[0]-1])
h5txt_file = open('h5txt_path.txt', 'w')
h5txt_file.write('HDF512_train.h5')
h5txt_file.close()
f.close()


