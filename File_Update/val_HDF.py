import  h5py
import numpy as np
import cv2 as cv

img_size=128
f = h5py.File('HDFbig_test.h5','r')
img_data=f['data'][:]
lab_data=f['label'][:]
freq_data=f['freq'][:]

# print(img_data)
for i in range(len(img_data)):

    print(img_data[i].transpose((2,1,0)).shape)
    print(lab_data[i].shape)
    print(freq_data[i].shape)

    cv.imshow('imgxx', img_data[i]*255)
    cv.waitKey()
    imgxx = (img_data[i].transpose((1,2,0))*255).astype(np.uint8)
    imgxx =  imgxx[:, :, [2, 1, 0]]
    imgc =imgxx.copy()


    for j in range(14):
        x = int(freq_data[i][2 * j] * img_size)
        y = int(freq_data[i][2 * j + 1] * img_size)
        print(x,y)
        imgc = cv.circle(imgc, (x,y), 2, (0, 255, 0), -1)


    cv.imshow('img',imgc)
    cv.waitKey()
