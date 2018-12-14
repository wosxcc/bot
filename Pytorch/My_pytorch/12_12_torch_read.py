from Pytorch.Torch_Model import SENet
import os
import  numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms,utils,datasets
from MY_Function.cv_function import dram_chinese
import cv2 as cv
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = getattr(SENet,'se_resnet_18')(num_classes = 2)
model.load_state_dict(torch.load('seNet18_96_weight.pkl'))
model = model
# print(model)
path_img = 'D:/bot/my_tf/img/val/cat'
count_cat =0
counts =0
start_time = datetime.datetime.now()
for imfile in os.listdir(path_img):
    # print(imfile)
    img = cv.imread(path_img + '/' + imfile)
    img = cv.resize(img, (112, 112), cv.INTER_CUBIC)

    # img_ = (np.array(img[:, :, ::-1].transpose((2, 0, 1)), dtype=np.float32) - 127.5) / 128
    img_ = (np.array(img[:, :, ::-1].transpose((2, 0, 1)), dtype=np.float32) - 0) / 255
    img_tensor = Variable(torch.from_numpy(np.resize(img_, [1, 3, 112, 112])))
    befored= datetime.datetime.now()
    output = model(img_tensor)
    print('检测时间为：',datetime.datetime.now() - befored,'\t',output[0])
    name = '狗'
    if output[0][0] > output[0][1]:
        name = '猫'
        count_cat += 1
    img = dram_chinese(img, '这是： ' + name, 0, 20)
    # cv.imshow('img',img)
    # cv.waitKey()
    # cv.destroyAllWindows()
    counts+=1
    if counts==500:
        break
print('耗时:', start_time, datetime.datetime.now(), datetime.datetime.now() - start_time)
print('有{0}只猫:{1}，有{2}只狗:{3}。'.format(count_cat, float(count_cat / counts), counts - count_cat, float(1.0 - count_cat / counts)))