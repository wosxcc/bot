import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.python.framework import graph_util
from face_ID_net.read_image import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
label_lines = []
image_lines = []


# 画坐标
def draw_form(MAX_STEP):
    step = MAX_STEP / 10
    img_H = 1000
    img_W = 1200
    coordinate = np.zeros((img_H, img_W, 3), np.uint8)
    coordinate[:, :, :] = 255
    line_c = 8
    coordinate = cv.line(coordinate, (100, img_H - 100), (img_W, img_H - 100), (0, 0, 0), 2)
    coordinate = cv.line(coordinate, (100, 0), (100, img_H - 100), (0, 0, 0), 2)

    for i in range(11):
        coordinate = cv.line(coordinate, (i * 100 + 100, img_H - 100), (i * 100 + 100, 0), (0, 0, 0), 1)
        coordinate = cv.line(coordinate, (100, i * 100 + 100), (img_W, i * 100 + 100), (0, 0, 0), 1)
        if i > 0:
            cv.putText(coordinate, str(i * step), (i * 100 + 100 - 32, img_H - 100 + 50), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 0, 0), 2)
        biaohao = '%.1f' % (1.0 - i * 0.1 - 0.2)
        if biaohao == '-0.0':
            cv.putText(coordinate, '0', (100 - 50, i * 100 + 100 + 10 + 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv.putText(coordinate, biaohao, (100 - 50, i * 100 + 100 + 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return coordinate
# 画点
def drow_spot(img,x,y,MAX_STEP):
    # for i in range(x.shape[0]):
    put_str='step:%d  loss:%.5f'%(x,y)
    # print(put_str)
    img[120:180,505:950,:]=255
    cv.putText(img, put_str,(500,150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    spot_x = max(int(x/MAX_STEP*1000+100),0)
    spot_y =max(int(900-y*1000),0)
    # print('画点位置：',spot_x,spot_y)
    cv.circle(img,(spot_x,spot_y),3,(0,0,255),-1)
    cv.imshow('LOSS',img)
    cv.waitKey(10)

def face_net(batch_size,height, width, n_classes,learning_rate=0.001,margin=5,run_train=True):
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
        W5 = weight_variable([3, 3, 256, 128])
        b5 = bias_variable([128])
        conv5 = tf.nn.conv2d(relu4, W5, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, b5), name='relu5')

    with tf.variable_scope('conv7') as scope:
        W7 = weight_variable([3, 3, 128, 256])
        b7= bias_variable([256])
        conv7 = tf.nn.conv2d(relu5, W7, strides=[1, 1, 1, 1], padding='SAME')
        relu7 = tf.nn.relu(tf.nn.bias_add(conv7, b7), name='relu7')

        # 全连接层
    with tf.variable_scope("fc1") as scope:
        dim = int(np.prod(relu7.get_shape()[1:]))
        reshape = tf.reshape(relu7, [-1, dim])
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
        loss = tf.reduce_mean(loss) #+ tf.reduce_mean(d_pos)
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


def run_training(txt_name):
    imgs = draw_form(MAX_STEP)
    logs_train_dir = './face72/faceID/'
    X_data = read_image()
    graph= face_net(BATCH_SIZE, IMG_H,IMG_W, N_CLASSES,learning_rate,margin=10,run_train=True)
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    y_step=0
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(global_step)
        y_step = int(float(global_step))
    loss_list ={}
    loss_list['x']=[]
    loss_list['y'] = []

    for step in np.arange(MAX_STEP):
        loss_avg = 0.0
        for i in range(BATCH_SIZE):
            xb= (step%200)*32+i
            _, tra_loss, sess_pos, sess_neg = sess.run([graph['optimize'],graph['loss'],graph['d_pos'],graph['d_neg']],feed_dict={
                        graph['x']: np.reshape(X_data[xb], (3, 64, 64, 3))})
            loss_avg+=tra_loss

        avg_loss =loss_avg/BATCH_SIZE
        loss_list['x'].append(step+y_step)
        loss_list['y'].append(avg_loss)
        loss_list['x'].append(step+y_step)
        loss_list['y'].append(tra_loss)
        drow_spot(imgs,step, tra_loss, MAX_STEP)
        print('同:',sess_pos,'不同:',sess_neg,'距离差',sess_neg-sess_pos)
        if step % 50 == 0:
            # print('同一个人',sess.run(tf.reduce_mean(sess_pos)),'\t',sess_pos)
            # print('不同一个人',sess.run(tf.reduce_mean(sess_neg)),'\t',sess_neg)
            print('Step %d,train loss = %.5f' % (step+y_step, tra_loss))
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                       ['output/output'])
            with tf.gfile.FastGFile(logs_train_dir + 'face72.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            # 每迭代50次，打印出一次结果
        if step % 200 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step+y_step)
            # 每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用
    sess.close()



txt_name= 'trains.txt'
IMG_W = 64
IMG_H = 64

BATCH_SIZE = 32
CAPACITY = 32
MAX_STEP = 40000
learning_rate = 0.00001
N_CLASSES = 128
run_training(txt_name)



