import tensorflow as tf
import numpy as np



# xdata=np.arange(-100,100,0.5,dtype=float)
xdata=tf.Variable(0,dtype=tf.float32)
ydata =pow(xdata,2) -18*xdata +52


def get_weight(shape,lambd):
    var =tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambd)(var))
    return var

optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(ydata)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(optimizer)
    print(sess.run(xdata))

