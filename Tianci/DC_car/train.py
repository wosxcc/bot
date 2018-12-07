import numpy as np
import tensorflow as tf
import csv
from math import *
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
flie_path = "my_train.csv"
cvs_read = csv.reader(open(flie_path))
xdata = np.zeros((50000,5816+12+31+24+60+2),dtype=np.float64)
ydata = np.zeros((50000,2),dtype=np.float64)


counts = 0
for lista in cvs_read:
    # print(len(lista))
    xdata[counts, int(lista[0])] = 1
    xdata[counts, 5816 + int(lista[1])] = 1
    xdata[counts, 5816 + 12 + int(lista[2])] = 1
    xdata[counts, 5816 + 12 + 31 + int(lista[3])] = 1
    xdata[counts, 5816 + 12 + 31 + 24 + int(lista[4])] = 1
    xdata[counts, -2] = float(lista[14])/180.0
    xdata[counts, -1] = float(lista[15])/180.0
    ydata[counts, 0] = float(lista[16])/180.0
    ydata[counts, 1] = float(lista[17])/180.0
    counts += 1
    if counts==50000:
        break
def getDistance(latA, lonA, latB, lonB):
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    try:
        pA = atan(rb / ra * tan(radLatA))
        pB = atan(rb / ra * tan(radLatB))
        x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
        c1 = (sin(x) - x) * (sin(pA) + sin(pB)) ** 2 / cos(x / 2) ** 2
        c2 = (sin(x) + x) * (sin(pA) - sin(pB)) ** 2 / sin(x / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (x + dr)
        return distance  # meter
    except:
        return 0.0000001


lr = tf.Variable(0.001,trainable=False)
lrx = tf.constant(0.95,dtype=tf.float32,name="lrx")
ra = tf.constant(6378140,dtype=tf.float64,name="ra")
rb = tf.constant(6356755,dtype=tf.float64,name="rb")
pi = tf.constant(0.017453292519943295769236907*180,dtype=tf.float64,name="mpi")
def weight_variable(shape,name='weight'):
    initial = tf.truncated_normal_initializer(stddev=0.0001,dtype=tf.float64)
    return tf.get_variable(name,shape=shape,dtype=tf.float64,initializer=initial)

def bias_variable(shape,name='biases'):
    initial =tf.constant_initializer(0.1)
    return tf.get_variable(name,shape=shape,dtype=tf.float64,initializer=initial)
X = tf.placeholder(tf.float64, shape=[None,5945], name='input')
Y = tf.placeholder(tf.float64, shape=[None,2], name='ydata')

with tf.variable_scope("fc1") as scope:
    weights1 = weight_variable([5945, 1024])   ##24*24*256*256
    biases1 = bias_variable([1024])
    fc1 = tf.nn.relu(tf.matmul(X, weights1) + biases1,name='fc1')
with tf.variable_scope("fc2") as scope:
    weights2 = weight_variable([1024, 512])  ##24*24*256*256
    biases2 = bias_variable([512])
    fc2 = tf.nn.relu(tf.matmul(fc1, weights2) + biases2, name='fc2')

with tf.variable_scope("fc3") as scope:
    weights3 = weight_variable([512, 256])  ##24*24*256*256
    biases3 = bias_variable([256])
    fc3 = tf.nn.relu(tf.matmul(fc2, weights3) + biases3, name='fc3')
with tf.variable_scope("fc4") as scope:
    weights4 = weight_variable([256, 128])  ##24*24*256*256
    biases4 = bias_variable([128])
    fc4 = tf.nn.relu(tf.matmul(fc3, weights4) + biases4, name='fc4')

with tf.variable_scope("fc5") as scope:
    weights5 = weight_variable([128, 2])  ##24*24*256*256
    biases5 = bias_variable([2])
    fc5 = tf.nn.relu(tf.matmul(fc4, weights5) + biases5, name='fc5')

    loss = tf.reduce_sum(tf.square(Y - fc5), name='loss')
    # latA = tf.slice(fc5, [0,0], [500,1])
    # lonA = tf.slice(fc5, [0,1], [500,1])
    # latB = tf.slice(Y, [0,0], [500,1])
    # lonB = tf.slice(Y, [0,1], [500,1])
    #
    # radLatA = latA * pi
    # radLonA = lonA * pi
    # radLatB = latB * pi
    # radLonB = lonB * pi
    #
    # pA = tf.atan(rb / ra * tf.tan(radLatA))
    # pB = tf.atan(rb / ra * tf.tan(radLatB))
    # x = tf.acos(tf.sin(pA) * tf.sin(pB) + tf.cos(pA) * tf.cos(pB) * tf.cos(radLonA - radLonB))
    # c1 = (tf.sin(x) - x) * (tf.sin(pA) + tf.sin(pB)) ** 2 / tf.cos(x / 2) ** 2
    # c2 = (tf.sin(x) + x) * (tf.sin(pA) - tf.sin(pB)) ** 2 / tf.sin(x / 2) ** 2
    # dr = ((ra - rb) / ra) / 8 * (c1 - c2)
    # # distance = ra * (x + dr)
    # loss =ra * (x + dr)

with tf.name_scope("optimizer"):
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
echo = 10000





with tf.Session()as sess:
    logs_train_dir = './object/1121/'
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    for step in range(echo+1):
        sumloss=0
        for i in range(100):
            # print(np.reshape(xdata[i*500:(i+1)*500], (500, 5945)).shape)
            # print(xdata[i*500],ydata[i*500])
            output,_,ycont ,zy= sess.run([loss,train_op,fc5,Y],
                                      feed_dict={X:np.reshape(xdata[i*500:(i+1)*500], (500, 5945)),Y:np.reshape(ydata[i*500:(i+1)*500], (500, 2))})
            #print("第",i ,"次loss:",np.mean(output)) # ,ycont[0],"看看有什么",yxa[0],yxb[0],mxa[0],mxb[0]
            print("第",i ,"次loss:","距离差",getDistance(ycont[0][0],ycont[0][1],zy[0][0],zy[0][1]))
            sumloss+=output
        print( "当前loss", sumloss / 100)
        if step%10==0:
            lr = lrx * lr
            checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
            saver.save(sess,checkpoint_path,global_step = step)
            print("当前学习率",sess.run(lr),"当前loss",sumloss/100)



# 第 92 次loss: 0.7002786261792839
# 第 93 次loss: 2.45752312812824
# 第 94 次loss: 1.9388613376509727
# 第 95 次loss: 3.5464445075475166
# 第 96 次loss: 1.2151866242223366
# 第 97 次loss: 3.6632534202291738
# 第 98 次loss: 1.768225841065837
# 第 99 次loss: 1.9005176112384157
#
# 第 92 次loss: 0.694183270866802
# 第 93 次loss: 2.4459200274146924
# 第 94 次loss: 1.9280361832277557
# 第 95 次loss: 3.543623676260706
# 第 96 次loss: 1.2113747554837644
# 第 97 次loss: 3.6571961581496697
# 第 98 次loss: 1.7656067851135733
# 第 99 次loss: 1.9002697068502366

