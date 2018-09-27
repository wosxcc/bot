import tensorflow as tf
import numpy as np
from keras.layers.merge import add,concatenate
from keras.layers import UpSampling2D
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def face_net(batch_size,height, width, n_classes,learning_rate,margin,image_count=3):
    print(batch_size,height, width, n_classes,learning_rate)
    x = tf.placeholder(tf.float32, shape=[batch_size,image_count, height, width, 3], name='input')
    # y = tf.placeholder(tf.float32, shape=[None, n_classes], name='labels')

    def weight_variable(shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)



    for xx in range(image_count):
        print('进入方法',xx)
        xnow_x =tf.slice(x, [0, xx, 0, 0, 0], [batch_size, 1,height, width, 3])
        now_x = tf.reshape(xnow_x, shape=[batch_size, height, width, 3], name=None)
        with tf.variable_scope('conv1') as scope:
            W1 = weight_variable([3, 3, 3, 32])
            b1 = bias_variable([32])
            conv = tf.nn.conv2d(now_x, W1, strides=[1, 1, 1, 1], padding="SAME")
            pre_activation = tf.nn.bias_add(conv, b1)
            relu1 = tf.tanh(pre_activation, name="tanh1")

        with tf.variable_scope('conv2') as scope:
            W2 = weight_variable([3, 3, 32, 64])
            b2 = bias_variable([64])
            conv2 = tf.nn.conv2d(relu1, W2, strides=[1, 2, 2, 1], padding='SAME')
            relu2 = tf.tanh(tf.nn.bias_add(conv2, b2), name='tanh2')


        with tf.variable_scope('conv3') as scope:
            W3 = weight_variable([3, 3, 64, 128])
            b3 = bias_variable([128])
            conv3 = tf.nn.conv2d(relu2, W3, strides=[1, 1, 1, 1], padding='SAME')
            relu3 = tf.tanh(tf.nn.bias_add(conv3, b3), name='tanh3')

        with tf.variable_scope('conv4') as scope:
            W4 = weight_variable([3, 3, 128, 256])
            b4 = bias_variable([256])
            conv4 = tf.nn.conv2d(relu3, W4, strides=[1, 2, 2, 1], padding='SAME')
            relu4 = tf.tanh(tf.nn.bias_add(conv4, b4), name='tanh4')


        with tf.variable_scope('conv5') as scope:
            W5 = weight_variable([3, 3, 256, 128])
            b5 = bias_variable([128])
            conv5 = tf.nn.conv2d(relu4, W5, strides=[1, 1, 1, 1], padding='SAME')
            relu5 = tf.tanh(tf.nn.bias_add(conv5, b5), name='tanh5')


        # with tf.variable_scope('conv6') as scope:
        #     W6 = weight_variable([3, 3, 512, 256])
        #     b6 = bias_variable([256])
        #     conv6 = tf.nn.conv2d(relu5, W6, strides=[1, 2, 2, 1], padding='SAME')
        #     relu6 = tf.nn.relu(tf.nn.bias_add(conv6, b6), name='relu6')



        # relu66 = relu3  +UpSampling2D(2)(relu5)

        relu66 = concatenate([ relu3, UpSampling2D(2)(relu5)])
        # print('看看是什么',relu66)
        with tf.variable_scope('conv7') as scope:
            W7 = weight_variable([3, 3, 256, 128])
            b7= bias_variable([128])
            conv7 = tf.nn.conv2d(relu66, W7, strides=[1, 2, 2, 1], padding='SAME')
            relu7 = tf.tanh(tf.nn.bias_add(conv7, b7), name='tanh7')


            # 全连接层
        with tf.variable_scope("fc1") as scope:
            dim = int(np.prod(relu7.get_shape()[1:]))
            reshape = tf.reshape(relu7, [-1, dim])
            weights1 =weight_variable([dim, 300])
            biases1 = bias_variable([300])
            fc1 = tf.tanh(tf.matmul(reshape, weights1) + biases1, name="fc1")

        with tf.variable_scope("output") as scope:
            weights2 = weight_variable([300, n_classes])
            biases2 = bias_variable([n_classes])
            y_conv = tf.add(tf.matmul(fc1, weights2), biases2, name="output")
            # y_conv =tf.tanh(y_conv,name="output")
        if xx == 0:
            anchor_out= y_conv
        elif xx == 1:
            positive_out= y_conv
        elif xx == 2:
            negative_out= y_conv

    if image_count==3:
        # d_pos = tf.reduce_sum(tf.square(anchor_out - positive_out), 1)
        d_pos = tf.norm(anchor_out - positive_out, axis=1)
        print('搞什么毛线d_pos',d_pos)
        # d_neg = tf.reduce_sum(tf.square(anchor_out - negative_out), 1)

        d_neg = tf.norm(anchor_out - negative_out, axis=1)
        # print('搞什么毛线', d_pos - d_neg)
        # print('瞎几把搞',margin + d_pos - d_neg)
        loss = tf.maximum(0.0, margin + d_pos - d_neg)
        print('你这是干什么',loss)
        loss = tf.reduce_mean(loss) + tf.reduce_mean(d_pos)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return dict(
            x=x,
            loss=loss,
            optimize=train_op,
            d_pos=d_pos,
            d_neg=d_neg,
        )
    if image_count==1:
        return dict(
            x=x,
            anchor_out=anchor_out,
        )