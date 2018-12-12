import cv2 as cv
import numpy as np
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
from MY_Function.cv_function import dram_chinese
import datetime
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)       # 3 表示输入图片是三通道  ，6表示6个卷积核 ， 5表示卷积核的大小             ----结果（batch, 6 imgW-4 ,imgH-4）  (batch,6,92,92)
        self.pool = nn.MaxPool2d(2,2)       # 2 表示池化核为大小2   ， 2 表示步长为2  (batch,6,46,46)
        self.conv2 = nn.Conv2d(6,16,5)      # ---(batch,16,42,42)
        self.fc1 = nn.Linear(16*21*21,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, 2)


    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*21*21)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net().cuda()

#
# data_transform = transforms.Compose(
#     [transforms.Resize(96),
#     transforms.CenterCrop(96),          # 图像裁剪成96*96大小
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]
# )
#
# test_data = datasets.ImageFolder(root='D:/bot/my_tf/img/val/',transform=data_transform)
path_img = 'D:/bot/my_tf/img/val/cat'
count_cat =0
start_time = datetime.datetime.now()
for imfile in os.listdir(path_img):

    img =cv.imread(path_img+'/'+imfile)
    img =cv.resize(img,(96,96),cv.INTER_CUBIC)


    net.load_state_dict(torch.load('params.pkl'))
    img_ = (np.array(img[:, :, ::-1].transpose((2, 0, 1)),dtype=np.float32)-127.5)/128
    # print(img_.shape)
    # print(np.resize(img_,[1,3,96,96]).shape)
    # print(torch.from_numpy(np.resize(img_,[1,3,96,96],)))
    img_tensor = Variable(torch.from_numpy(np.resize(img_, [1, 3, 96, 96]))).cuda()
    output = net(img_tensor)
    name = '狗'
    if output[0][0] > output[0][1]:
        name = '猫'
        count_cat += 1
    img = dram_chinese(img, '这是： ' + name, 0, 20)
    # cv.imshow('img',img)
    # cv.waitKey()
    # cv.destroyAllWindows()
print('耗时:',start_time,datetime.datetime.now(),datetime.datetime.now()-start_time)
print('有{0}只猫:{1}，有{2}只狗:{3}。'.format(count_cat, float(count_cat / 110), 110 - count_cat, float(1.0 - count_cat / 110)))
