import numpy as np
import tensorflow as tf
import csv
from math import *
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
flie_path = "my_train.csv"
cvs_read = csv.reader(open(flie_path))
xdata = np.zeros((50000, 5816 + 12 + 31 + 24 + 60 + 2), dtype=np.float64)
ydata = np.zeros((50000, 2), dtype=np.float64)

counts = 0
for lista in cvs_read:
    # print(len(lista))
    xdata[counts, int(lista[0])] = 1
    xdata[counts, 5816 + int(lista[1])] = 1
    xdata[counts, 5816 + 12 + int(lista[2])] = 1
    xdata[counts, 5816 + 12 + 31 + int(lista[3])] = 1
    xdata[counts, 5816 + 12 + 31 + 24 + int(lista[4])] = 1
    xdata[counts, -2] = float(lista[14])/180
    xdata[counts, -1] = float(lista[15])/180
    ydata[counts, 0] = float(lista[16])/180
    ydata[counts, 1] = float(lista[17])/180
    counts += 1
    if counts == 50000:
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


ra = tf.constant(6378140, dtype=tf.float64, name="ra")
rb = tf.constant(6356755, dtype=tf.float64, name="rb")
pi = tf.constant(0.017453292519943295769236907*180, dtype=tf.float64, name="mpi")


def weight_variable(shape, name='weight'):
    initial = tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float64)
    return tf.get_variable(name, shape=shape, dtype=tf.float64, initializer=initial)


def bias_variable(shape, name='biases'):
    initial = tf.constant_initializer(0.1)
    return tf.get_variable(name, shape=shape, dtype=tf.float64, initializer=initial)


X = tf.placeholder(tf.float64, shape=[None, 5945], name='input')
Y = tf.placeholder(tf.float64, shape=[None, 2], name='ydata')






weights1 = tf.Variable(tf.random_normal([5945,256]),dtype=tf.float64,name = 'weights1')
bias1 = tf.Variable(tf.zeros([256]),dtype=tf.float64,name = 'bias1')
acc = tf.nn.sigmoid(tf.matmul(X,weights1) + bias1)
weights2 = tf.Variable(tf.random_normal([256,2]),dtype=tf.float64,name = 'weights2')
bias2 = tf.Variable(tf.zeros([2]),dtype=tf.float64,name = 'bias2')
fc2 = tf.nn.sigmoid(tf.matmul(acc,weights2) + bias2)

# with tf.variable_scope("fc1") as scope:
#     weights1 = weight_variable([5945, 256])  ##24*24*256*256
#     biases1 = bias_variable([256])
#     fc1 = tf.nn.relu(tf.matmul(X, weights1) + biases1, name='fc1')
# with tf.variable_scope("fc2") as scope:
#     weights2 = weight_variable([256, 2])  ##24*24*256*256
#     biases2 = bias_variable([2])
#     fc2 = tf.nn.relu(tf.matmul(fc1, weights2) + biases2, name='fc2')

latA = tf.slice(fc2, [0, 0], [500, 1])
lonA = tf.slice(fc2, [0, 1], [500, 1])
latB = tf.slice(Y, [0, 0], [500, 1])
lonB = tf.slice(Y, [0, 1], [500, 1])

radLatA = latA * pi
radLonA = lonA * pi
radLatB = latB * pi
radLonB = lonB * pi

pA = tf.atan(rb / ra * tf.tan(radLatA))
pB = tf.atan(rb / ra * tf.tan(radLatB))
x = tf.acos(tf.sin(pA) * tf.sin(pB) + tf.cos(pA) * tf.cos(pB) * tf.cos(radLonA - radLonB))
c1 = (tf.sin(x) - x) * (tf.sin(pA) + tf.sin(pB)) ** 2 / tf.cos(x / 2) ** 2
c2 = (tf.sin(x) + x) * (tf.sin(pA) - tf.sin(pB)) ** 2 / tf.sin(x / 2) ** 2
dr = ((ra - rb) / ra) / 8 * (c1 - c2)
# distance = ra * (x + dr)
loss = ra * (x + dr)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

echo = 100

with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(echo):
        for i in range(100):
            # print(np.reshape(xdata[i*500:(i+1)*500], (500, 5945)).shape)
            # print(xdata[i*500],ydata[i*500])
            output, _, ycont, yxa, yxb, mxa, mxb = sess.run([loss, train_op, fc2, latA, lonA, latB, lonB],
                                                            feed_dict={X: np.reshape(xdata[i * 500:(i + 1) * 500],
                                                                                     (500, 5945)),
                                                                       Y: np.reshape(ydata[i * 500:(i + 1) * 500],
                                                                                     (500, 2))})
            # print(output)
            print("第", i, "次loss:", np.mean(output), yxa[0], yxb[0], mxa[0], mxb[0])




