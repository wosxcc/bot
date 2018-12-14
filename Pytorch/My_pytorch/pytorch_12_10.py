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
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_transform = transforms.Compose(
    [transforms.Resize(96),
    transforms.CenterCrop(96),          # 图像裁剪成96*96大小
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]
)

train_data = datasets.ImageFolder(root='D:/bot/my_tf/img/train/',transform=data_transform)
# print('训练数据',train_data)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=4,
                                           shuffle=True,
                                           num_workers= 4)

test_data = datasets.ImageFolder(root='D:/bot/my_tf/img/val/',transform=data_transform)
test_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=4,
                                           shuffle=True,
                                           num_workers= 4)


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
net = Net()
net.load_state_dict(torch.load('params.pkl'))
if __name__ == '__main__':
    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr =0.0001,momentum=0.9)


    for epoch in range(2):
        run_loss = 0.0
        # print(train_loader)
        for i,data in enumerate(train_loader,0):
            # print(i,data)
            inputs ,labels = data
            print('看看',labels)
            inputs ,labels = Variable(inputs),Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            print(labels)
            loss  = cirterion(outputs,labels)
            loss.backward()
            optimizer.step()


            # print(loss.data)
            run_loss += loss.data.numpy()
            if i %2000 == 1999:
                torch.save(net, 'model.pkl')                 # 保存整个网络，包括整个计算图
                torch.save(net.state_dict(), 'params.pkl')  #  只保存网络中的参数 (速度快, 占内存少)
                print('[%d %5d] loss：%.3f'%(epoch+1,i+1,run_loss/i))
