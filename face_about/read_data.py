import numpy as np
import os
import cv2 as cv

import random


def rdata():
    txt_path = 'E:/about_Face/faceID1'
    img_path = 'E:/about_Face/faceID'
    x_data = []
    y_data = []
    countc=0
    # face_init[-3] 性别 -1，0，1
    # face_init[-4] 微笑 0，1，2
    # face_init[-2] 眼镜 0，1，2
    # face_init[-6] 年龄 -1 0-100
    # face_init[150] 得分
    # face_init[6:150] 关键点
    for txt_flie in os.listdir(txt_path):
        # if countc>15000:
        if txt_flie[:-6] + '.jpg' in os.listdir(img_path):
            img = cv.imread(img_path + '/' + txt_flie[:-6] + '.jpg')
        else:
            img = cv.imread(img_path + '/' + txt_flie[:-6] + '.png')
        face_init = open(txt_path+"/"+txt_flie).read().split(' ')

        a_data = np.zeros([156],np.float)

        # a_data[0] = int(float(face_init[-3])) + 1
        # a_data[1] = int(float(face_init[-4]))
        # a_data[2] + int(float(face_init[-2]))

        a_data[int(float(face_init[-3]))+1]=1
        a_data[3+int(float(face_init[-4]))] = 1
        a_data[6 + int(float(face_init[-2]))] = 1
        if float(face_init[-6])==-1 or float(face_init[-6])==0:
            a_data[9] = 0
        else:
            a_data[9] = 1
            a_data[10] = float(face_init[-6])/100
        a_data[11]=float(face_init[150])
        x_rand = 0
        y_rand = 0
        if img.shape[0]==182:
            x_rand = random.randint(0, 11)
            y_rand = random.randint(0, 11)
            img= img[y_rand:182-y_rand,x_rand:182-x_rand,:]
        for i in  range(72):
            a_data[12 + 2 * i] = (float(face_init[6+2*i])-x_rand)/img.shape[0]
            a_data[12 + 2 * i + 1] = (float(face_init[6+2*i+1]) - x_rand) / img.shape[1]

        img = cv.resize(img, (160, 160), cv.INTER_CUBIC)
        #a_data[12:] =np.array(face_init[6:150],dtype=np.float)/img.shape[0]

        # print(a_data[:12],a_data[12:14])
        x_data.append(img)
        y_data.append(a_data)

        # for i in range(72):
        #     cv.circle(img,(int(float(a_data[12+2*i])*img.shape[1]),int(float(a_data[12+2*i+1])*img.shape[0])),2,(255,0,0),-1)

        # cv.imshow('img',img)
        # cv.waitKey()
        # cv.destroyAllWindows()
        #
        # if countc==128:
        #     break
        countc+=1


    x_data = ((np.array(x_data,dtype=np.float32)-127.5)/128)
    y_data = (np.array(y_data,np.float)-0.5)*2
    return x_data,y_data



def rdata_file():
    txt_path = 'E:/about_Face/faceID1'
    img_path = 'E:/about_Face/faceID'
    x_data = []
    y_data = []
    countc=0

    out_put = ''
    for txt_flie in os.listdir(txt_path):
        if txt_flie[:-6] + '.jpg' in os.listdir(img_path):

            out_put +=  img_path + '/' + txt_flie[:-6] + '.jpg'+'---'
            x_data.append(img_path + '/' + txt_flie[:-6] + '.jpg')
        else:
            out_put += img_path + '/' + txt_flie[:-6] + '.png' + '---'
            x_data.append(img_path + '/' + txt_flie[:-6] + '.png')
        y_data.append(txt_path+"/"+txt_flie)
        out_put += txt_path+"/"+txt_flie + '\n'
        print(txt_flie)


    txt_save=open('train_file.txt','w')
    txt_save.write(out_put)
    txt_save.close()
    return x_data,y_data




def file_to_data(x_data,y_data):
    x_read =[]
    y_read =[]

    for i in range(len(x_data)):
        img = cv.imread(x_data[i])


        face_init = open(y_data[i]).read().split(' ')

        a_data = np.zeros([156], np.float)
        a_data[int(float(face_init[-3])) + 1] = 1       # 性别
        a_data[3 + int(float(face_init[-4]))] = 1       # 表情
        a_data[6 + int(float(face_init[-2]))] = 1       # 眼镜
        if float(face_init[-6]) == -1 or float(face_init[-6]) == 0:     # 年龄
            a_data[9] = 0
        else:
            a_data[9] = 1
            a_data[10] = float(face_init[-6]) / 100
        a_data[11] = float(face_init[150])              #得分
        x_rand = 0
        y_rand = 0
        if img.shape[0] == 182:
            x_rand = random.randint(0, 11)
            y_rand = random.randint(0, 11)
            img = img[y_rand:182 - y_rand, x_rand:182 - x_rand, :]
        for i in range(72):
            a_data[12 + 2 * i] = (float(face_init[6 + 2 * i]) - x_rand) / img.shape[0]
            a_data[12 + 2 * i + 1] = (float(face_init[6 + 2 * i + 1]) - x_rand) / img.shape[1]

        img = cv.resize(img, (160, 160), cv.INTER_CUBIC)

        x_read.append(img)
        y_read.append(a_data)
    x_read = ((np.array(x_read, dtype=np.float32) - 127.5) / 128)
    y_read = (np.array(y_read, np.float) - 0.5) * 2
    return x_read, y_read


def val_train(X_data,Y_data):
    max_batch = 8
    img_m = np.zeros((960, 960, 3), np.uint8)

    for xb in range(64):
        imgc = X_data[xb] * 128 + 127.5
        imga = np.zeros((imgc.shape), np.uint8)
        imga[:, :, :] = imgc

        for xxx in range(int((156 - 12) / 2)):
            nx = int((Y_data[xb][xxx * 2 + 2] + 1) / 2 * 160)
            ny = int((Y_data[xb][xxx * 2 + 3] + 1) / 2 * 160)
            cv.circle(imga, (nx, ny), 2, (0, 255, 255), -1)
        imga = cv.resize(imga, (120, 120), interpolation=cv.INTER_CUBIC)
        # cv.imshow('imga', imga)
        # cv.waitKey()
        img_m[xb // 8 * 120:xb // 8 * 120 + 120, xb % 8 * 120:xb % 8 * 120 + 120, :] = imga

    cv.imshow('img_m', img_m)
    cv.waitKey()
    # cv.destroyAllWindows()
# # rdata_file()
# X_data_flie=open('train_file.txt').read().split('\n')
# batch_img = []
# batch_lab = []
# for xi in range(64):
#     xxx = random.randint(0, len(X_data_flie) - 1)
#     # print(xxx)
#     batch_img.append(X_data_flie[xxx].split('---')[0])
#     batch_lab.append(X_data_flie[xxx].split('---')[1])
# print(len(batch_img))
# # print(batch_img)
# # print(batch_lab)
# mxdata,mydata =file_to_data(batch_img,batch_lab)
# print(mxdata.shape)
# print(mydata.shape)
# val_train(mxdata,mydata)