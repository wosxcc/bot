import torch
from torch.autograd import Variable
import  numpy #as np

x = torch.Tensor(5,3)
print(x)

x2 = torch.rand(5,3)
x3 = torch.rand(5,3)
print(x2.size(),x2)
print('x2+3',x2+x3)  #形状相同才能相加


print(x)

# print('x2+3',torch.add(x+x)) #出错


a = x2.numpy() # 转化为numpy数组
print(a)




# 创建一个变量
x = Variable(torch.ones(2,2),requires_grad=True)


y = x +2
print(y)
print(x)

z = y*y*y+2
print('z',z)
out = z.mean()  # 平均值

print('out',out)