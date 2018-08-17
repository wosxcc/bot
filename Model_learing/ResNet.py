# coding:utf-8
import collections
import tensorflow as tf
import time
import math
import os
import numpy as np
import cv2 as cv
import datetime
from datetime import datetime
from tensorflow.python.framework import graph_util


slim = tf.contrib.slim






class Block(collections.namedtuple('Bolck', ['scope', 'unit_fn', 'args'])):
    'A named tuple describing a ResNet block.'


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
    ximage_lines/=255

    xlabel_linesc=np.array(label_linesc, dtype='float32')
    return ximage_lines,xlabel_linesc



def subsample(inputs, factor, scope = None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride = factor, scope = scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope = None):
                # 输入数据，抽取特征数，卷积核大小， 步长

    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride = 1, padding = 'SAME', scope = scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride = stride, padding = 'VALID', scope = scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections = None):
    print(blocks)
    #   Block(scope='block1', unit_fn=<function add_arg_scope.<locals>.func_with_args at 0x000001FBF28C6158>, args=[(128, 32, 1), (128, 32, 1), (128, 32, 2)])
    # , Block(scope='block2', unit_fn=<function add_arg_scope.<locals>.func_with_args at 0x000001FBF28C6158>, args=[(256, 64, 1), (256, 64, 1), (256, 64, 1), (256, 64, 2)])
    # , Block(scope='block3', unit_fn=<function add_arg_scope.<locals>.func_with_args at 0x000001FBF28C6158>, args=[(512, 128, 1), (512, 128, 1), (512, 128, 1), (512, 128, 1), (512, 128, 1), (512, 128, 2)])
    # , Block(scope='block4', unit_fn=<function add_arg_scope.<locals>.func_with_args at 0x000001FBF28C6158>, args=[(1024, 512, 1), (1024, 512, 1), (1024, 512, 1)])
    print('输入的net',net)

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                print('i',i)
                print('unit',unit)
                with tf.variable_scope('unit_%d' %(i + 1), values = [net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    # 抽取特征数量， 残差学习单元    , 步长
                    print(unit_depth, unit_depth_bottleneck, unit_stride )

                    # 调用blocks['unit_fn']（也就是bottleneck方法）残差网咯
                    net = block.unit_fn(net, depth = unit_depth,depth_bottleneck=unit_depth_bottleneck,stride = unit_stride)

                    print('看看什么是',net)
                    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
                    print('看完后是什么', net)
    print('返回的值',net)
    return net



# 定义残差网络
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections = None, scope = None):

    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank = 4)
        preact = slim.batch_norm(inputs, activation_fn = tf.nn.relu, scope = 'preact')

        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride = stride, normalizer_fn = None, activation_fn = None, scope = 'shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride = 1, scope = 'conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope = 'conv2')
        print('挖断上帝', residual)
        residual = slim.conv2d(residual, depth, [1, 1], stride = 1, normalizer_fn = None, activation_fn = None, scope = 'conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)



def resnet_v2(inputs, blocks, BATCH_SIZE, num_classes = None, global_pool = True, include_root_block = True, reuse = None, scope = None):
    # print('inputs, blocks, BATCH_SIZE',inputs, blocks, BATCH_SIZE)
    print(inputs)
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse = reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections = end_points_collection):
            net = inputs
        if include_root_block:
            with slim.arg_scope([slim.conv2d], activation_fn = None, normalizer_fn = None):
                # print('看看当前的net',net)
                net = conv2d_same(net, 64, 7, stride = 1, scope = 'conv1')
            # print('if里面的net',net)
            # net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'pool1')

        # print('进入算法前的net',net)
        net = stack_blocks_dense(net, blocks)

        net = slim.batch_norm(net, activation_fn = tf.nn.relu, scope = 'postnorm')
        if global_pool:
            net = tf.reduce_mean(net, [1, 2], name = 'pool5', keep_dims = True)
        if num_classes is not None:
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'logits')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope = 'predictions')
            net = tf.reshape(net, shape=[1, num_classes],name = 'output')
            return net, end_points



def resnet_v2_50(inputs,BATCH_SIZE, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_50'):
    # print(num_classes, global_pool, reuse)
    blocks = [
        Block('block1', bottleneck, [(128, 32, 1)] * 2 + [(128, 32, 2)]),
        Block('block2', bottleneck, [(256, 64, 1)] * 3 + [(256, 64, 2)]),
        Block('block3', bottleneck, [(512, 128, 1)] * 5 + [(512, 128, 2)]),
        Block('block4', bottleneck, [(1024, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks,BATCH_SIZE, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)

def losses(logits, labels):
    loss = tf.sqrt(tf.reduce_mean(tf.square(logits - labels)))
    return loss

def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy

def train_resnet():
    X_data, Y_data = read_img(txt_name)
    logs_train_dir ='./faceres/log/'
    x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_H, 3], name='input')
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES], name='labels')

    net, end_points = resnet_v2_50(x,BATCH_SIZE, num_classes=N_CLASSES)
    train_loss=losses(net, y)
    train_op=trainning(train_loss, learning_rate)
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
    for step in np.arange(MAX_STEP):
        for i in range(BATCH_SIZE):
            xb= (step%(int(len(X_data)/BATCH_SIZE)))*BATCH_SIZE+i
            _, xloss = sess.run([train_op,train_loss], feed_dict={
                x: np.reshape(X_data[xb], (1, 96, 96, 3)),
                y: np.reshape(Y_data[xb], (1, 30))})
        if step % 50 == 0:
            print('第%d批次，当前loss%.5f' % (step,xloss))



            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                       ['resnet_v2_50/output'])
            with tf.gfile.FastGFile(logs_train_dir + 'face72.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

        if step % 200 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
    sess.close()




init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
learning_rate=0.0000001
txt_name= 'trains.txt'
N_CLASSES = 30
IMG_W = 96
IMG_H = 96
BATCH_SIZE = 32
CAPACITY = 64
MAX_STEP = 1000000
train_resnet()



# def get_one_image(img_dir):
#     image = cv.imread(img_dir)
#
#     # 好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
#     min_bian=min(image.shape[0],image.shape[1])
#     max_bian = max(image.shape[0], image.shape[1])
#
#     if min_bian/max_bian<0.6:
#         bei_x = 48 / max_bian
#         if image.shape[0] == min_bian:
#             cha = int((image.shape[1] - min_bian) / 2)
#             images = np.zeros((image.shape[1], image.shape[1], 3), np.uint8)
#             images[cha:cha + min_bian, :, :] = image
#             image = cv.resize(images, None, fx=bei_x, fy=bei_x, interpolation=cv.INTER_CUBIC)
#         else:
#             cha = int((image.shape[0] - min_bian) / 2)
#             images = np.zeros((image.shape[0], image.shape[0], 3), np.uint8)
#             images[:, cha:cha + min_bian, :] = image
#             image = cv.resize(images, None, fx=bei_x, fy=bei_x, interpolation=cv.INTER_CUBIC)
#     else:
#         bei_x = 48 / min_bian
#         if image.shape[0]>min_bian:
#             cha=int((image.shape[0]-min_bian)/2)
#             image = cv.resize(image[cha:min_bian+cha,:], None, fx=bei_x, fy=bei_x, interpolation=cv.INTER_CUBIC)
#         else:
#             cha = int((image.shape[1] - min_bian) / 2)
#             image = cv.resize(image[:,cha:min_bian+cha], None, fx=bei_x, fy=bei_x, interpolation=cv.INTER_CUBIC)
#     # cv.imshow('image',image)
#     # cv.waitKey()
#     image_arr = np.array(image)
#     return image_arr
#
#
# def xtval(test_file):
#     log_dir = './resnet/log2/'
#     # image_arr=test_file
#     image_arr = get_one_image(test_file)
#     with tf.Graph().as_default():
#         image = tf.cast(image_arr, tf.float32)
#         image = tf.image.per_image_standardization(image)
#         image = tf.reshape(image, [1, 48, 48, 3])
#         p, end_points = resnet_v2_50(image,1,2)
#         # p = tf.reshape(p, shape=[1, -1])
#         logits = tf.nn.softmax(p)
#         x = tf.placeholder(tf.float32,shape = [48,48,3])
#         saver = tf.train.Saver()
#         with tf.Session() as sess:
#             ckpt = tf.train.get_checkpoint_state(log_dir)
#             if ckpt and ckpt.model_checkpoint_path:
#                 global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 #调用saver.restore()函数，加载训练好的网络模型
#                 print('Loading success')
#             else:
#                 print('No checkpoint')
#             prediction = sess.run(logits, feed_dict={x: image_arr})
#             return prediction
#
# pathsss='E:/xbot/face_into/face68/image_test'
# for test_file in os.listdir(pathsss):
#
#
#     xtime=datetime.now()
#     prediction = xtval(pathsss + '/' + test_file)
#     print('耗时:',datetime.now()-xtime)
#     max_index = np.argmax(prediction)
#     img =cv.imread(pathsss+'/'+test_file)
#
#     if max_index == 0:
#         print('是狗的概率是：', prediction[0][0])
#         cv.putText(img, 'dog:{}'.format(str(prediction[0][0])), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     elif max_index == 1:
#         cv.putText(img, 'cat:{}'.format(str(prediction[0][1])), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         print('是猫的概率是：', prediction[0][1])
#     cv.imshow('img', img)
#     cv.waitKey()




