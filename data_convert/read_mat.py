import cv2 as cv
import numpy as np
import os
import scipy.io as scio

mat_path='E:/xcc_download/hand_dataset/training_dataset/training_data/annotations'
img_path='E:/xcc_download/hand_dataset/training_dataset/training_data/images'

for file_name in os.listdir(mat_path):
    if file_name[-4:]=='.mat':
        data = scio.loadmat(mat_path+'/'+file_name)


        # print('看看矩形框',data['boxes'])
        img = cv.imread(img_path + '/' + file_name[:-4] + '.jpg')
        for bboxs in data['boxes']:
            # print('我去',bboxs)
            for bbox in bboxs:
                # print('哈哈哈',bbox)
                for bbox1 in bbox:
                    print('我的天', bbox1)
                    for bbox2 in bbox1:
                        print(int(bbox2[0][0][0]),int(bbox2[0][0][1]))
                        img = cv.circle(img, (int(bbox2[0][0][0]),int(bbox2[0][0][1])), 2, (0, 0, 255), -1)
                        img = cv.circle(img, (int(bbox2[1][0][0]),int(bbox2[1][0][1])), 2, (0, 0, 255), -1)
                        img = cv.circle(img, (int(bbox2[2][0][0]),int(bbox2[2][0][1])), 2, (0, 0, 255), -1)
                        img = cv.circle(img, (int(bbox2[3][0][0]),int(bbox2[3][0][1])), 2, (0, 0, 255), -1)
                        print('哈哈哈0', bbox2[0][0])
                        print('哈哈哈1', bbox2[1][0])
                        print('哈哈哈2', bbox2[2][0])
                        print('哈哈哈3', bbox2[3][0])



        cv.imshow('img',img)
        cv.waitKey()

