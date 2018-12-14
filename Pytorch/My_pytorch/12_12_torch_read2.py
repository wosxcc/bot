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


data_transform = transforms.Compose(
    [transforms.Resize(112),                            # 图像缩放
    transforms.RandomCrop(112),                         # 图像随机裁剪成96*96大小
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)
test_data = datasets.ImageFolder(root='D:/bot/my_tf/img/val/',transform=data_transform)
# print('训练数据',train_data)
test_loader = torch.utils.data.DataLoader(test_data,
                                           batch_size=1,           # 定义批次大小
                                           shuffle=True,            # 乱序操作
                                           num_workers= 1)          # 多少个子进程加载数据

model = getattr(SENet,'se_resnet_18')(num_classes = 2)
model.load_state_dict(torch.load('seNet18_96_weight.pkl'))
count_cat =0
counts=220
if __name__ == '__main__':
    correct = 0
    total = 0

    for data in test_loader:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('total：',total,'correct:',correct)


#
# for i,data in enumerate(test_loader,0):
#     inputs, labels = data
#
#     print(i,inputs, labels)
#     inputs, labels = Variable(inputs), Variable(labels)
#     outputs = model(inputs)
#     name = 1
#     if outputs[0][0] > outputs[0][1]:
#         name = 0
#
#     print(labels.data[0])
#     if name == labels.data[0]:
#         count_cat+=1
#
# print('有{0}只猫:{1}，有{2}只狗:{3}。'.format(count_cat, float(count_cat / counts), counts - count_cat, float(1.0 - count_cat / counts)))