import tensorflow as tf
import datetime
from tensorflow.python.framework import graph_util
from face_ID_net.read_image import *
def face_net(batch_size,height, width, n_classes,learning_rate=0.001,margin=0.3,run_train=True):
    x = tf.placeholder(tf.float32, shape=[None, height, width, 3], name='input')
    def weight_variable(shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    with tf.variable_scope('conv1') as scope:
        W1 = weight_variable([3, 3, 3, 32])
        b1 = bias_variable([32])
        conv = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, b1)
        relu1 = tf.nn.relu(pre_activation, name="relu1")

    with tf.variable_scope('conv2') as scope:
        W2 = weight_variable([3, 3, 32, 64])
        b2 = bias_variable([64])
        conv2 = tf.nn.conv2d(relu1, W2, strides=[1, 2, 2, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name='relu2')

    with tf.variable_scope('conv3') as scope:
        W3 = weight_variable([3, 3, 64, 128])
        b3 = bias_variable([128])
        conv3 = tf.nn.conv2d(relu2, W3, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name='relu3')

    with tf.variable_scope('conv4') as scope:
        W4 = weight_variable([3, 3, 128, 256])
        b4 = bias_variable([256])
        conv4 = tf.nn.conv2d(relu3, W4, strides=[1, 2, 2, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, b4), name='relu4')

    with tf.variable_scope('conv5') as scope:
        W5 = weight_variable([3, 3, 256, 512])
        b5 = bias_variable([512])
        conv5 = tf.nn.conv2d(relu4, W5, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, b5), name='relu5')

    with tf.variable_scope('conv6') as scope:
        W6 = weight_variable([3, 3, 512, 1024])
        b6 = bias_variable([1024])
        conv6 = tf.nn.conv2d(relu5, W6, strides=[1, 1, 1, 1], padding='SAME')
        relu6 = tf.nn.relu(tf.nn.bias_add(conv6, b6), name='relu6')

    with tf.variable_scope('conv7') as scope:
        W7 = weight_variable([3, 3, 1024, 256])
        b7= bias_variable([256])
        conv7 = tf.nn.conv2d(relu6, W7, strides=[1, 1, 1, 1], padding='SAME')
        relu7 = tf.nn.relu(tf.nn.bias_add(conv7, b7), name='relu7')

        # 全连接层
    with tf.variable_scope("fc1") as scope:
        dim = int(np.prod(relu7.get_shape()[1:]))
        reshape = tf.reshape(relu7, [-1, dim])
        print("看啊可能",dim)
        weights1 =weight_variable([dim, 256])   ##24*24*256*256
        biases1 = bias_variable([256])
        fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, weights1) + biases1, name="fc1"),0.5)

    with tf.variable_scope("fc2") as scope:
        weights122 =weight_variable([256, 1024])
        biases122 = bias_variable([1024])
        fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc1, weights122) + biases122, name="fc2"),0.5)

    with tf.variable_scope("output") as scope:
        weights2 = weight_variable([1024, n_classes])
        biases2 = bias_variable([n_classes])
        y_conv=tf.add(tf.matmul(fc2, weights2),biases2, name="output")

    if run_train==True:
        print('最后一层输出',y_conv)
        anchor_out = tf.slice(y_conv, [ 0, 0], [1, n_classes])
        positive_out =  tf.slice(y_conv, [ 1, 0], [1, n_classes])
        negative_out =  tf.slice(y_conv, [ 2, 0], [1, n_classes])
        d_pos = tf.norm(anchor_out - positive_out, axis=1)
        print('搞什么毛线d_pos', d_pos)
        # d_neg = tf.reduce_sum(tf.square(anchor_out - negative_out), 1)
        d_neg = tf.norm(anchor_out - negative_out, axis=1)
        loss = tf.maximum(0.0, margin + d_pos - d_neg)
        print('你这是干什么', loss)
        loss = tf.reduce_mean(loss)# + tf.reduce_mean(d_pos)
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
    else:
        anchor_out = tf.slice(y_conv, [0, 0], [1,n_classes])
        return dict(
            x=x,
            anchor_out=anchor_out,
        )
