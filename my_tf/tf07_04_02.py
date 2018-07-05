#!/usr/bin/python
# -*- coding:utf-8 -*-
import input_data as input
import tensorflow as tf
from tensorflow.python.framework import graph_util

#全连接神经网络训练手写字体
def testMnist():

    mnist = input.read_data_sets("/home/myjob/Downloads/Mnist/", one_hot=True)
    x = tf.placeholder("float32",[None,784])
    w = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,w)+b)
    y_ = tf.placeholder("float32",[None,10])
    cross_entry = tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entry)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            yy = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        print "over:"
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
        print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

    print mnist
#权重函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#获取常量
def bias_Variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return initial
#卷积函数
def conv2d(x,W):

    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
#最大池化函数
def padding_2d(x):

    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
#使用卷积网路训练手写字体
def mnist_conv2d():
   #读取数据
    mnist = input.read_data_sets("/home/myjob/Downloads/Mnist/", one_hot=True)
    x = tf.placeholder("float32", [None, 784],name='input_x')
    y_ = tf.placeholder("float32", [None, 10],name='input_y')

    w_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_Variable([32])
    x_image = tf.reshape(x,[-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
    h_pool1 = padding_2d(h_conv1)

    w_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_Variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)

    h_pool2 = padding_2d(h_conv2)

    w_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_Variable([1024])

    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1,name="fc1")

    keep_prob = tf.placeholder("float32",name='keep_prob')

    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    w_fc2 = weight_variable([1024,10])
    b_fc2 = bias_Variable([10])

    y_conv2d = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2,name="out")
  #交叉熵损失函数
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv2d))

   #优化函数

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  #计算网络精度

    correct_prediction = tf.equal(tf.argmax(y_conv2d, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    model_path_pb = "/home/myjob/Downloads/Mnist/"

    with tf.Session() as sess:
        sess.run(init)
        for i in range(101):
            batch = mnist.train.next_batch(64)
            if i % 2 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print ("step %d, training accuracy %g" % (i, train_accuracy))
                # saver_path = saver.save(sess,model_path)

               #保存成pb文件

                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['out'])

                with tf.gfile.FastGFile(model_path_pb +'model1.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

                result = sess.run(y_conv2d,feed_dict={x:mnist.test.images[0:1],keep_prob:1.0})
                index = tf.argmax(result,1)

                # print result
                # print sess.run(index)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            v = sess.graph.get_tensor_by_name('Variable_1:0')
            print ("========================================")
            print (sess.run(v[0]))

        print ("test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images[0:100], y_: mnist.test.labels[0:100], keep_prob: 1.0}))