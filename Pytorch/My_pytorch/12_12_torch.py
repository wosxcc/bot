from Pytorch.Torch_Model import SENet
import os
import  numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import random
from torchvision import transforms,utils,datasets
from Pytorch.My_pytorch.read_data import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# X_data_flie=open('dogcat.txt').read().split('\n')



data_transform = transforms.Compose(
    [transforms.Resize(112),                            # 图像缩放
    transforms.RandomCrop(112),                         # 图像随机裁剪成96*96大小
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)

train_data = datasets.ImageFolder(root='D:/bot/my_tf/img/train/',transform=data_transform)
# print('训练数据',train_data)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=64,           # 定义批次大小
                                           shuffle=True,            # 乱序操作
                                           num_workers= 4)          # 多少个子进程加载数据

test_data = datasets.ImageFolder(root='D:/bot/my_tf/img/val/',transform=data_transform)
test_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=4,
                                           shuffle=True,        # 乱序操作
                                           num_workers= 4)



model = getattr(SENet,'se_resnet_18')(num_classes = 2)

BATCH_SIZE=64
# model.load_state_dict(torch.load('seNet18_96_weight.pkl'))
model = model.cuda()
if __name__ == '__main__':
    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr =0.01,momentum=0.9)

    for epoch in range(15):
        run_loss = 0.0
        # print(train_loader)
        for i,data in enumerate(train_loader,0):
            # print(i,data)
            inputs ,labels = data
            # labels =labels
            # labels=labels*2-1
            # print(inputs)
            # print('看看',labels)
            inputs ,labels = Variable(inputs).cuda(),Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print(i,loss.data)
            run_loss += loss.cpu().data.numpy()
            if i % 150 == 149:
                torch.save(model, 'seNet18_96.pkl')  # 保存整个网络，包括整个计算图
                torch.save(model.state_dict(), 'seNet18_96_weight.pkl')  # 只保存网络中的参数 (速度快, 占内存少)
                print('[%d %5d] loss：%.3f' % (epoch + 1, i + 1, run_loss / i))