import os
import  cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import graph_util
def get_data_img():
    img_path = 'E:/Face_Data/lfwcrop_color/faces'
    faceids  ={}
    for file in os.listdir(img_path):
        if file[:-9] not in faceids:
            faceids[file[:-9]]=[]
        faceids[file[:-9]].append(file)
    face_img = []
    face_lab = []
    face_class = 0
    for name in faceids:
        for face_name in faceids[name]:
            face_img.append(cv.imread(img_path+'/'+face_name))
            face_lab.append(face_class)
        face_class+=1
    face_img = np.array(face_img,dtype='float32')
    face_lab = np.array(face_lab,dtype='int')
    return face_img,face_lab







def face_id_net(Batch_size,img_W,img_H,face_class,learning_rate,is_train=True):

    x = tf.placeholder(tf.float32,shape=[None,img_H,img_W,3],name='input')
    y = tf.placeholder(tf.float32,shape=[None,Batch_size],name='label')

    def batch_norm(x,is_train):
        name = 'batch_norm'

        with tf.variable_scope(name):
            is_train =tf.convert_to_tensor(is_train,dtype=tf.bool)
            n_out = int(x.get_shape()[-1])
            beta = tf.Variable(tf.constant(0.0,shape=[n_out],dtype= x.dtype),
                               name = name+'/beta',trainable=True,dtype=x.dtype)
            gamma = tf.Variable(tf.constant(1.0,shape=[n_out],dtype=x.dtype),
                                name=name+'/gamma',trainable=True,dtype=x.dtype)

            batch_mean,batch_var = tf.nn.moments(x,[0],name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.9)

            def mean_var_with_update():
                ema_apply_op =ema.apply([batch_mean,batch_var])

                with tf.control_dependencies([ema_apply_op]):
                    return  tf.identity(batch_mean),tf.identity(batch_var)
            mean,var = control_flow_ops.cond(is_train,
                                             mean_var_with_update,
                                             lambda:(ema.average(batch_mean)),ema.average(batch_var))
            normed = tf.nn.batch_normalization(x,mean,var,beta,gamma,1e-3)

        return normed


    def weight_variable(shape,name='weight'):
        initial = tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32)
        return tf.get_variable(name,shape=shape,dtype=tf.float32,initializer=initial)

    def bias_variable(shape,name='biases'):
        initial =tf.constant_initializer(0.1)
        return tf.get_variable(name,shape=shape,dtype=tf.float32,initializer=initial)


    with tf.variable_scope('conv1') as scope:
        W1 = weight_variable([11, 11, 3, 32])
        b1 = bias_variable([32])
        conv = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, b1)
        relu1 = batch_norm(tf.nn.relu(pre_activation, name="relu1"),is_train)

    with tf.variable_scope('conv2') as scope:
        W2 = weight_variable([3, 3, 32, 64])
        b2 = bias_variable([64])
        conv2 = tf.nn.conv2d(relu1, W2, strides=[1, 2, 2, 1], padding='SAME')
        relu2 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv2, b2), name='relu2'),is_train)

    with tf.variable_scope('conv3') as scope:
        W3 = weight_variable([9, 9, 64, 16])
        b3 = bias_variable([16])
        conv3 = tf.nn.conv2d(relu2, W3, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv3, b3), name='relu3'),is_train)

    with tf.variable_scope('conv4') as scope:
        W4 = weight_variable([9, 9, 16, 32])
        b4 = bias_variable([32])
        conv4 = tf.nn.conv2d(relu3, W4, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv4, b4), name='relu4'),is_train)

    with tf.variable_scope('conv5') as scope:
        W5 = weight_variable([3, 3, 32, 256])
        b5 = bias_variable([256])
        conv5 = tf.nn.conv2d(relu4, W5, strides=[1, 2, 2, 1], padding='SAME')
        relu5 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv5, b5), name='relu5'),is_train)

    with tf.variable_scope('conv6') as scope:
        W6 = weight_variable([7, 7, 256, 32])
        b6 = bias_variable([32])
        conv6 = tf.nn.conv2d(relu5, W6, strides=[1, 1,1, 1], padding='SAME')
        relu6 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv6, b6), name='relu6'),is_train)

    with tf.variable_scope('conv7') as scope:
        W7 = weight_variable([5, 5, 32, 32])
        b7= bias_variable([32])
        conv7 = tf.nn.conv2d(relu6, W7, strides=[1, 1, 1, 1], padding='SAME')
        relu7 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv7, b7), name='relu7'),is_train)

        # 全连接层
    with tf.variable_scope("fc1") as scope:
        dim = int(np.prod(relu7.get_shape()[1:]))
        reshape = tf.reshape(relu7, [-1, dim])
        weights1 =weight_variable([dim, 1024])   ##24*24*256*256
        biases1 = bias_variable([1024])
        fc1 = batch_norm(tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, weights1) + biases1, name="fc1"),0.5),is_train)

    # with tf.variable_scope("fc2") as scope:
    #     weights122 =weight_variable([1024, 1024])
    #     biases122 = bias_variable([1024])
    #     fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc1, weights122) + biases122, name="fc2"),0.5)

    with tf.variable_scope("output") as scope:
        weights2 = weight_variable([1024, face_class])
        biases2 = bias_variable([face_class])
        y_conv=tf.add(tf.matmul(fc1, weights2),biases2, name="output")

    





