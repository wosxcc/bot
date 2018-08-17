# coding:utf-8
import tensorflow as tf
import numpy as np
import cv2 as cv
from keras.layers.merge import add,concatenate
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers import UpSampling2D
from face_into.DenseNet_face.subpixel import SubPixelUpscaling
import  os
from tensorflow.python.framework import  graph_util
import collections
slim = tf.contrib.slim

label_lines = []
image_lines = []

def read_img(txt_name):
    label_lines = []
    image_lines = []
    txt_open = open(txt_name)
    txt_read = txt_open.read()
    txt_lines = txt_read.split('\n')

    for line in txt_lines:
        xlabel = []
        if len(line)>3:
            line_list = line.split(' ')
            image_lines.append(cv.imread(line_list[0]))
            xlabel.append(line_list[1])
            xlabel.append(line_list[2])
            for x in range(14):
                xlabel.append(line_list[117 + 2 + x * 2])
                xlabel.append(line_list[117 + 2 + x * 2 + 1])
            label_lines.append(xlabel)

            # label_lines.append(line_list[1:])

    label_linesc=[[float(i) for i in xline] for xline in label_lines]
    ximage_lines=np.array(image_lines, dtype='float32')
    ximage_lines=ximage_lines/256   # -127.5
    xlabel_linesc=np.array(label_linesc, dtype='float32')
    return ximage_lines,xlabel_linesc


def conv2d_block(x,weight_number, bottleneck= False,dropout_rate=None):
    # tf.nn.l2_normalize   归一化借鉴
    x = tf.nn.relu(x)
    if bottleneck:
        weightx4 = weight_number*4
        x = slim.conv2d(x, weightx4, 3, stride=1, padding='same')
        x = tf.nn.relu(x)

    x =  slim.conv2d(x, weight_number, 3, stride=1, padding='same')

    if dropout_rate:
        x = tf.nn.dropout(x,dropout_rate)
    return  x

def dense_blosk(x,number_layer ,nb_filter ,weight_number,bottleneck= False,dropout_rate=None, grow_nb_filters=True, return_concat_list=False ):

    x_list = [x]


    for  i in range(number_layer):
        cd = conv2d_block(x ,weight_number,bottleneck,dropout_rate)

        x_list.append(x)
        x = concatenate([ x, cd])

        if grow_nb_filters:
            nb_filter += weight_number

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter

def transition_up_block(x, nb_filters, type='deconv'):
    if type =='upsampling':
        x = UpSampling2D(2)(x)
    elif type== 'subpixel':
        x = slim.conv2d(x,nb_filters,3,1,padding='SAME')
        x = tf.nn.relu(x)
        s =SubPixelUpscaling(scale_factor=2)(x)
        x = slim.conv2d(x,nb_filters,3,1,padding='SAME')
        x = tf.nn.relu(x)
    else:
        x = tf.nn.conv2d_transpose(x, 3, output_shape=[-1, -1, -1, nb_filters],
                                   strides=[1, 2, 2, 1], padding="SAME")
    return  x

def transition_block(x ,nb_filters,compression=1.0):
    x = tf.nn.relu(x)

    x = slim.conv2d(x,int(nb_filters*compression),1,1,padding='SAME')
    x = tf.nn.avg_pool(x,2,2)

    return x

def dense_net(nclass, img_input,include_top ,depth =121 ,np_dense_block = 4,weight_number =32,nb_filter =-1,
               nb_layers_per_block = -1,bottleneck = False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False):

    if type(nb_layers_per_block) is list or   type(nb_layers_per_block) is tuple:
        np_layers = list(nb_layers_per_block)
        assert len(np_layers) == (np_dense_block)

        final_nb_layer = np_layers[-1]
        np_layers = np_layers[:-1]

    compression = 1.0 -reduction

    if nb_filter <=0:
        nb_filter = 2*weight_number

    if subsample_initial_block:
        initial_kernel =7
        initial_strides =2
    else:
        initial_kernel = 3
        initial_strides = 1

    x = slim.conv2d(img_input,weight_number,initial_kernel,initial_strides,padding = 'same')

    if subsample_initial_block:
        x=tf.nn.relu(x)
        x= tf.nn.max_pool( x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")


    for block_idx in range(np_dense_block -1):
        x, nb_filter = dense_blosk(x, np_layers[block_idx], nb_filter, weight_number
                                   , bottleneck=bottleneck, dropout_rate=dropout_rate)
        x = transition_block(x,nb_filter,compression=compression)
        nb_filter = int(nb_filter * compression)

    x ,nb_filter =dense_blosk(x,final_nb_layer,nb_filter,weight_number
                              ,bottleneck=bottleneck, dropout_rate=dropout_rate )

    x = tf.nn.relu(x)

    if include_top:
        x =Dense(nclass,activation='softmax')(x) #通过activation激活函数等把
    return x