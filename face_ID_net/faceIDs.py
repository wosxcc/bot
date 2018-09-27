import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.python.framework import graph_util
# from keras.layers.merge import add,concatenate
# from keras.layers import UpSampling2D
from face_ID_net.IDnet import  face_net
from face_ID_net.read_image  import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
label_lines = []
image_lines = []

# def face_net(batch_size,height, width, n_classes,learning_rate,margin,image_count=3):
#     print(batch_size,height, width, n_classes,learning_rate)
#     x = tf.placeholder(tf.float32, shape=[batch_size,3, height, width, 3], name='input')
#     # y = tf.placeholder(tf.float32, shape=[None, n_classes], name='labels')
#
#     def weight_variable(shape, name="weights"):
#         initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
#         return tf.Variable(initial, name=name)
#
#     def bias_variable(shape, name="biases"):
#         initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
#         return tf.Variable(initial, name=name)
#
#
#
#     for xx in range(image_count):
#         print('进入方法',xx)
#         xnow_x =tf.slice(x, [0, xx, 0, 0, 0], [batch_size, 1,height, width, 3])
#         now_x = tf.reshape(xnow_x, shape=[batch_size, height, width, 3], name=None)
#         with tf.variable_scope('conv1') as scope:
#             W1 = weight_variable([3, 3, 3, 32])
#             b1 = bias_variable([32])
#             conv = tf.nn.conv2d(now_x, W1, strides=[1, 1, 1, 1], padding="SAME")
#             pre_activation = tf.nn.bias_add(conv, b1)
#             relu1 = tf.nn.relu(pre_activation, name="relu1")
#
#         with tf.variable_scope('conv2') as scope:
#             W2 = weight_variable([3, 3, 32, 64])
#             b2 = bias_variable([64])
#             conv2 = tf.nn.conv2d(relu1, W2, strides=[1, 2, 2, 1], padding='SAME')
#             relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name='relu2')
#
#
#         with tf.variable_scope('conv3') as scope:
#             W3 = weight_variable([3, 3, 64, 128])
#             b3 = bias_variable([128])
#             conv3 = tf.nn.conv2d(relu2, W3, strides=[1, 1, 1, 1], padding='SAME')
#             relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name='relu3')
#
#         with tf.variable_scope('conv4') as scope:
#             W4 = weight_variable([3, 3, 128, 256])
#             b4 = bias_variable([256])
#             conv4 = tf.nn.conv2d(relu3, W4, strides=[1, 2, 2, 1], padding='SAME')
#             relu4 = tf.nn.relu(tf.nn.bias_add(conv4, b4), name='relu4')
#
#
#         with tf.variable_scope('conv5') as scope:
#             W5 = weight_variable([3, 3, 256, 128])
#             b5 = bias_variable([128])
#             conv5 = tf.nn.conv2d(relu4, W5, strides=[1, 1, 1, 1], padding='SAME')
#             relu5 = tf.nn.relu(tf.nn.bias_add(conv5, b5), name='relu5')
#
#
#         # with tf.variable_scope('conv6') as scope:
#         #     W6 = weight_variable([3, 3, 512, 256])
#         #     b6 = bias_variable([256])
#         #     conv6 = tf.nn.conv2d(relu5, W6, strides=[1, 2, 2, 1], padding='SAME')
#         #     relu6 = tf.nn.relu(tf.nn.bias_add(conv6, b6), name='relu6')
#
#
#
#         # relu66 = relu3  +UpSampling2D(2)(relu5)
#
#         relu66 = concatenate([ relu3, UpSampling2D(2)(relu5)])
#         # print('看看是什么',relu66)
#         with tf.variable_scope('conv7') as scope:
#             W7 = weight_variable([3, 3, 256, 128])
#             b7= bias_variable([128])
#             conv7 = tf.nn.conv2d(relu66, W7, strides=[1, 2, 2, 1], padding='SAME')
#             relu7 = tf.nn.relu(tf.nn.bias_add(conv7, b7), name='relu7')
#
#
#             # 全连接层
#         with tf.variable_scope("fc1") as scope:
#
#             dim = int(np.prod(relu7.get_shape()[1:]))
#             reshape = tf.reshape(relu7, [-1, dim])
#             weights1 =weight_variable([dim, 300])
#             biases1 = bias_variable([300])
#             fc1 = tf.nn.relu(tf.matmul(reshape, weights1) + biases1, name="fc1")
#
#         with tf.variable_scope("output") as scope:
#             weights2 = weight_variable([300, n_classes])
#             biases2 = bias_variable([n_classes])
#             y_conv = tf.add(tf.matmul(fc1, weights2), biases2, name="output")
#             y_conv =tf.tanh(y_conv,name="output")
#         if xx == 0:
#             anchor_out= y_conv
#         elif xx == 1:
#             positive_out= y_conv
#         elif xx == 2:
#             negative_out= y_conv
#
#     if image_count==3:
#         # d_pos = tf.reduce_sum(tf.square(anchor_out - positive_out), 1)
#         d_pos = tf.norm(anchor_out - positive_out, axis=1)
#         print('搞什么毛线d_pos',d_pos)
#         # d_neg = tf.reduce_sum(tf.square(anchor_out - negative_out), 1)
#
#         d_neg = tf.norm(anchor_out - negative_out, axis=1)
#         # print('搞什么毛线', d_pos - d_neg)
#         # print('瞎几把搞',margin + d_pos - d_neg)
#         loss = tf.maximum(0.0, margin + d_pos - d_neg)
#         print('你这是干什么',loss)
#         loss = tf.reduce_mean(loss) + tf.reduce_mean(d_pos)
#         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#         global_step = tf.Variable(0, name="global_step", trainable=False)
#         train_op = optimizer.minimize(loss, global_step=global_step)
#
#         return dict(
#             x=x,
#             loss=loss,
#             optimize=train_op,
#             d_pos=d_pos,
#             d_neg=d_neg,
#         )
#     if image_count==1:
#         return dict(
#             x=x,
#             anchor_out=anchor_out,
#         )



def run_training(txt_name):
    logs_train_dir = './face72/faceIDcard/'

    X_data = read_image()
    # print(X_data.shape)
    graph= face_net(BATCH_SIZE, IMG_H,IMG_W, N_CLASSES,learning_rate,15,3)
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('读取模型成功')
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(global_step)

    for step in np.arange(MAX_STEP):
        # for i in range(BATCH_SIZE):
        #     xb= (step%400)*16+i
            # ximage=np.array(X_data[xb]*255, dtype='uint8')
            # for xxi in range(72):
            #     cv.circle(ximage,(int(Y_data[xb][2+2*xxi]*96),int(Y_data[xb][2+2*xxi+1]*96)),2,(0, 255, 255), -1)
            # cv.imshow('ximage',ximage)
            # cv.waitKey()

        _ ,tra_loss,sed_pos,sed_neg= sess.run([graph['optimize'],graph['loss'],graph['d_pos'],graph['d_neg']],feed_dict={
                    graph['x']: np.reshape(X_data[(step%400)*16:(step%400)*16+16], (16,3, 64, 64, 3))
                    })#,graph['y']: np.reshape(Y_data[xb], (1, 30))

        # print(se_anchor,se_positive,se_anchor)
        # print('se_anchor',se_anchor)
        # print('se_positive', se_positive)
        # print('se_negative', se_negative)
        if step % 50 == 0:
            print('同一个人',sed_pos,sess.run(tf.reduce_mean(sed_pos)))
            print('不同一个人', sed_neg,sess.run(tf.reduce_mean(sed_neg)))
            print('Step %d,train loss = %.5f' % (step, tra_loss))
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                       ['output/output'])
            with tf.gfile.FastGFile(logs_train_dir + 'face72.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())


            # 每迭代50次，打印出一次结果
            # summary_str = sess.run(summary_op)
            # train_writer.add_summary(summary_str, step)
        if step % 200 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            # 每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用
    sess.close()



txt_name= 'trains.txt'
IMG_W = 64
IMG_H = 64

BATCH_SIZE = 16
CAPACITY = 16
MAX_STEP = 6000
learning_rate = 0.0001
N_CLASSES = 128
run_training(txt_name)


# def get_one_image(img_dir):
#     image = cv.imread(img_dir)
#     # 好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
#     bei_x = 64 / int(image.shape[1])
#     bei_y = 64 / int(image.shape[0])
#     min_bian = min(image.shape[0], image.shape[1])
#     max_bian = max(image.shape[0], image.shape[1])
#     # bei_x = 48 / max_bian
#     # print(12346)
#     # if image.shape[0] == min_bian:
#     #     cha = int((image.shape[1] - min_bian) / 2)
#     #     images = np.zeros((image.shape[1], image.shape[1], 3), np.uint8)
#     #     images[cha:cha + min_bian, :, :] = image
#     #     image = cv.resize(images, None, fx=bei_x, fy=bei_x, interpolation=cv.INTER_CUBIC)
#     # else:
#     #     cha = int((image.shape[0] - min_bian) / 2)
#     #     images = np.zeros((image.shape[0], image.shape[0], 3), np.uint8)
#     #     images[:, cha:cha + min_bian, :] = image
#     #     image = cv.resize(images, None, fx=bei_x, fy=bei_x, interpolation=cv.INTER_CUBIC)
#     image = cv.resize(image, None, fx=bei_x, fy=bei_y, interpolation=cv.INTER_CUBIC)
#     image_arr = np.array(image)
#
#     return image_arr


def val(image_arr):
    log_dir = './face72/faceIDcard/'
    # image_arr=test_file
    with tf.Graph().as_default():
        # image = tf.cast(image_arr, tf.float32)
        # image = tf.image.per_image_standardization(image)  ###归一化操作
        # image = tf.reshape(image_arr, [1,3, 64, 64, 3])
        # print('图片是什么',image)
        graph = face_net(1, IMG_H,IMG_W, N_CLASSES,learning_rate,15,3)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 调用saver.restore()函数，加载训练好的网络模型
                # print('Loading success')
            else:
                print('没有保存的模型')
            pos_d,neg_d = sess.run([graph['d_pos'],graph['d_neg']], feed_dict={graph['x']: np.reshape(image_arr, (1,3, 64, 64, 3))})
            return pos_d,neg_d

X_data = read_image()


for i in range(200):

    start_time = datetime.datetime.now()
    pos_d, neg_d = val(X_data[6400+i])
    print('耗时：',datetime.datetime.now()-start_time     )
    img = (X_data[6400+i][0]*256 +128).astype(np.uint8)
    img = cv.resize(img, (480, 480), interpolation=cv.INTER_CUBIC)
    print( pos_d,neg_d)
    # biaoq ='None'
    # if prediction[0][0]>= 0.8 and prediction[0][0]<1.6:
    #     biaoq = 'Smile'
    # elif prediction[0][0]>=1.6:
    #     biaoq = 'Laugh'
    # biaoq+=':' + str(prediction[0][1])
    # img = cv.putText(img, biaoq, (0, 30), 2, cv.FONT_HERSHEY_PLAIN, (255, 0, 0))
    # for i in range(int(len(prediction[0]) / 2)-1):
    #     cv.circle(img, (int(prediction[0][2+i * 2] * img.shape[1]), int(prediction[0][2+i * 2 + 1] * img.shape[0])), 2,
    #               (0, 255, 255), -1)

    cv.imshow('img', img)
    cv.waitKey()
    cv.destroyAllWindows()