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




# net.load_state_dict(torch.load('params.pkl'))
def save():
    net = torch.nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.MaxPool2d(2, 2),
        nn.ReLU(),
        nn.Conv2d(6, 16, 5),
        nn.MaxPool2d(2, 2),
        nn.ReLU(),

    )
    net2 = torch.nn.Sequential(
        nn.Linear(16 * 21 * 21, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )

    cirterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    optimizer2 = optim.SGD(net2.parameters(),lr =0.0001,momentum=0.9)

    net.load_state_dict(torch.load('params_net1.pkl'))
    net2.load_state_dict(torch.load('params_net2.pkl'))
    print('网络是可以了')
    for epoch in range(5):
        run_loss = 0.0
        print('进入大循环')
        for i,data in enumerate(train_loader,0):
            # print(i,data)
            inputs ,labels = data
            inputs ,labels = Variable(inputs),Variable(labels)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            outputs = net2(net(inputs).view(-1, 16 * 21 * 21))

            loss  = cirterion(outputs,labels)
            loss.backward()
            optimizer1.step()
            optimizer2.step()


            # print(loss.data)
            run_loss += loss.data.numpy()
            if i %2000 == 1999:
                torch.save(net, 'model_net1.pkl')                 # 保存整个网络，包括整个计算图
                torch.save(net2, 'model_net2.pkl')
                torch.save(net.state_dict(), 'params_net1.pkl')  #  只保存网络中的参数 (速度快, 占内存少)
                torch.save(net2.state_dict(), 'params_net2.pkl')
                print('[%d %5d] loss：%.3f'%(epoch+1,i+1,run_loss/i))
if __name__ == '__main__':


    save()




 # model = nn.Sequential(
 #                  nn.Conv2d(1,20,5),
 #                  nn.ReLU(),
 #                  nn.Conv2d(20,64,5),
 #                  nn.ReLU()
 #                )
 #
 #        # Example of using Sequential with OrderedDict
 #        model = nn.Sequential(OrderedDict([
 #                  ('conv1', nn.Conv2d(1,20,5)),
 #                  ('relu1', nn.ReLU()),
 #                  ('conv2', nn.Conv2d(20,64,5)),
 #                  ('relu2', nn.ReLU())
 #                ]))
