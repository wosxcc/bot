import numpy as np
import  random
import matplotlib.pyplot as plt
batch = 20
xdata = 100
h = 100
ydata = 10


x = np.random.uniform(-20,20, size=(batch, xdata))

xb = np.random.normal(0.9, 1, size=(batch, xdata))

y = x*0.7 #+ xb

w = 0.33
b = 0.33
learning_rate  = 0.01
for step in range(10000):
    y_cont =x*w+b
    loss = np.square(y_cont-y).sum()

    grad_y_pred = 2.0*(y_cont-y)
    grad_w =y_cont.T.dot(grad_y_pred)

    print(y_cont.shape)
    print("次数",step,"损失值",loss,grad_w.shape)
    w -=np.mean(grad_w)*learning_rate
    print(w)
plt.scatter(x, y, marker = 'o',color = 'red', s = 2 ,label = 'First')



plt.show()
print("输入值",x)
print("偏差值",xb)
print("真实值",y)



# w1 = np.random.randn(xdata,H)