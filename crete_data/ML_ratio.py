import numpy as np
import tensorflow as tf
import matplotlib.pyplot  as plt
import os
import datetime


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
txt_data  = open('boxs.txt','r')
txt_read = txt_data.read()
txt_lines = txt_read.split('\n')

ydata =[]
xdata =[]



for line in txt_lines:
    # print('第一个',line)
    line_split =line.split(' ')
    y = float(line_split[2])

    x= float(line_split[1])
    if float(line_split[0])-y/2.0 >0.02 and float(line_split[0])-y/2.0 <0.98  and float(line_split[3])/y>1.1:
        ydata.append(y)
        xdata.append(x)

hsf= np.array(xdata)*0.225-0.021


hsf2= np.array(xdata)*0.225-0.021+0.065

hsf3= np.array(xdata)*0.225-0.021-0.065
plt.figure()
plt.scatter(xdata,ydata)


print(hsf)
plt.plot(xdata,hsf,'r-',lw=3)
plt.plot(xdata,hsf2,'g-',lw=3)
plt.plot(xdata,hsf3,'b-',lw=3)
plt.show()

print('ydata',len(ydata))
print('xdata',len(xdata))


X = tf.placeholder(tf.float32,name ='X')
Y = tf.placeholder(tf.float32, name = 'Y')

W = tf.Variable(np.random.rand(),name = "weight")
B = tf.Variable(np.random.rand(),name = "bias")

Y_per = tf.add(tf.multiply(X,W),B)

tloass = tf.reduce_mean(tf.pow((Y-Y_per),2))/2
optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(tloass)


sess =tf.Session()

sess.run(tf.global_variables_initializer())
stime =datetime.datetime.now()
for step in range(50000):

    _ ,ttloss,xw,xb = sess.run([optimizer,tloass,W,B],
                               feed_dict = {X:xdata,Y:ydata})
    if step %100 == 0:
        print('第%d次训练，loss值是：%.5f,权重是：%.5f,偏移量是：%.5f，'% (step,ttloss,xw,xb))






sess.close()
print('耗时：',datetime.datetime.now()-stime)

#单GPU：0耗时:   0:02:22.004555
#单GPU：1耗时：  0:02:17.534709
#双GPU耗时：     0:02:24.310592