import numpy as np
import cv2 as cv
import torch
import os
import torch.utils.data.dataloader
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torchvision import transforms,utils,datasets
import torch.nn.functional as F
import torch.optim as optim
import warnings
import shutil
import collections
from torch.autograd import Variable
import datetime

from MY_Function.cv_function import dram_chinese

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

before = datetime.datetime.now()
net1 =torch.load('model_net1.pkl')
net2 =torch.load('model_net2.pkl')
print('数据转换耗时',before,'\t',datetime.datetime.now(),'\t',datetime.datetime.now()-before)
path_img = 'D:/bot/my_tf/img/val/dog'
count_cat =0
start_time = datetime.datetime.now()
for imfile in os.listdir(path_img):

    img =cv.imread(path_img+'/'+imfile)
    img =cv.resize(img,(96,96),cv.INTER_CUBIC)
    img_ = (np.array(img[:, :, ::-1].transpose((2, 0, 1)), dtype=np.float32) - 127.5) / 128. #
    img_tensor = Variable(torch.from_numpy(np.resize(img_, [1, 3, 96, 96]))).float()
    # #

    # img_ = np.array(img[:, :, ::-1].transpose((2, 0, 1)), dtype=np.float32)/ 255.0
    # img_ =np.resize(img_, [1, 3, 96, 96])
    # img_ = torch.from_numpy(img_).float()
    # img_tensor = Variable(img_)

    # before =datetime.datetime.now()

    # print('数据转换耗时',before,'\t',datetime.datetime.now(),'\t',datetime.datetime.now()-before)


    output = net2(net1(img_tensor).view(-1, 16 * 21 * 21))
    name = '狗'
    if output[0][0]>output[0][1]:
        name= '猫'
        count_cat+=1
    img =dram_chinese(img,'这是： '+name,0,20)
    # cv.imshow('img',img)
    # cv.waitKey()
    # cv.destroyAllWindows()
print('耗时:',start_time,datetime.datetime.now(),datetime.datetime.now()-start_time)
print('有{0}只猫:{1}，有{2}只狗:{3}。'.format(count_cat,float(count_cat/110),110-count_cat,float(1.0-count_cat/110)))


