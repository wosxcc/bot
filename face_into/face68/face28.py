import os
import io
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

# from image_read import readimg

position = []
image_payh = []
txt_name = 'face6.txt'


def read_text(txt_name):
    file_txt = open('trainb.txt')
    get = file_txt.read()
    result = get.split('\n')
    other_result = get.splitlines()
    try:
        for i in range(len(other_result)):
            data_file = result[i].split(' ')
            xdata=[]
            image_payh.append('E:/face68/train/' + data_file[0])
            for i in range(int(len(data_file[1:]) / 4)):
                xdata.append(data_file[1+4 * i])
                xdata.append(data_file[1+4 * i + 1])
            # print(xdata)
            position.append(xdata)
    except:
        print('Erro')
    positionc = [[float(i) for i in line] for line in position]  ##字符串转数字
    return image_payh, positionc


def get_batch(image, position, img_W, img_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    input_queue = tf.train.slice_input_producer([image, position])
    position = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # image = tf.image.resize_images(image, [img_W, img_H], method=0)

    image = tf.image.resize_image_with_crop_or_pad(image, img_W, img_H)
    image = tf.image.random_brightness(image, max_delta=0.5)  ##在-0.5到0.5之间随机调整亮度
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  ###在-0.5到0.5之间随机调整亮度
    image = tf.image.random_hue(image, 0.5)  ##在0-0.5之间随机调整图像饱和度
    image = tf.image.per_image_standardization(image)
    image_batch, position_batch = tf.train.batch([image, position],
                                                 batch_size=batch_size,
                                                 num_threads=32,
                                                 capacity=capacity)

    position_batch = tf.reshape(position_batch, [batch_size, 56])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, position_batch


def inference(images, y_data, batch_size, n_classes):
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    ##第一层卷积
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 32],  ###卷积核宽高，通道数，以及每个卷积核抽取的特征数量
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')

        ##第二层卷积
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 32, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
        ##第三层卷积
    with tf.variable_scope("conv3") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 64, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm2, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name="conv3")

        # pool3 && norm3
    with tf.variable_scope("pooling3_lrn") as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling3")
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm3')

    ##第四层卷积
    with tf.variable_scope("conv4") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 128, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm3, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name="conv4")

        # pool4 && norm4
    with tf.variable_scope("pooling3_lrn") as scope:
        pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling3")
        norm4 = tf.nn.lrn(pool4, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm4')

    # 全连接层
    with tf.variable_scope("fc1") as scope:
        reshape = tf.reshape(norm4, shape=[batch_size, -1])
        # print('reshape',reshape)
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    # full_connect2
    with tf.variable_scope("fc2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")

    # softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        y_conv = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
        rmse = tf.sqrt(tf.reduce_mean(tf.square(y_data - y_conv)))
    return y_conv, rmse


def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def run_training(txt_name):
    logs_train_dir = './face_line/log3/'
    train, train_into = read_text(txt_name)
    train_batch, train_into_batch = get_batch(train, train_into,
                                              IMG_W,
                                              IMG_H,
                                              BATCH_SIZE,
                                              CAPACITY)

    train_logits, rmse = inference(train_batch, train_into_batch, BATCH_SIZE, N_CLASSES)
    train_op = trainning(rmse, learning_rate)
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss = sess.run([train_op, rmse])
            if step % 50 == 0:
                print('Step %d,train loss = %.5f' % (step, tra_loss))
                # 每迭代50次，打印出一次结果
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                # 每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用

    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def get_one_image(img_dir):
    image = cv.imread(img_dir)
    # 好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
    bei_x = 96 / int(image.shape[1])
    bei_y = 96 / int(image.shape[0])
    min_bian = min(image.shape[0], image.shape[1])
    max_bian = max(image.shape[0], image.shape[1])
    # bei_x = 48 / max_bian
    # print(12346)
    # if image.shape[0] == min_bian:
    #     cha = int((image.shape[1] - min_bian) / 2)
    #     images = np.zeros((image.shape[1], image.shape[1], 3), np.uint8)
    #     images[cha:cha + min_bian, :, :] = image
    #     image = cv.resize(images, None, fx=bei_x, fy=bei_x, interpolation=cv.INTER_CUBIC)
    # else:
    #     cha = int((image.shape[0] - min_bian) / 2)
    #     images = np.zeros((image.shape[0], image.shape[0], 3), np.uint8)
    #     images[:, cha:cha + min_bian, :] = image
    #     image = cv.resize(images, None, fx=bei_x, fy=bei_x, interpolation=cv.INTER_CUBIC)
    image = cv.resize(image, None, fx=bei_x, fy=bei_y, interpolation=cv.INTER_CUBIC)
    image_arr = np.array(image)

    return image_arr


def val(test_file):
    log_dir = './face_line/log3/'
    # image_arr=test_file
    image_arr = get_one_image(test_file)
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)  ###归一化操作
        image = tf.reshape(image, [1, 96, 96, 3])
        op_intp = np.zeros(56, np.float32)
        p, r = inference(image, op_intp, 1, 56)
        # print('看看p的值：',p)
        logits = p  # tf.nn.softmax(p)
        x = tf.placeholder(tf.float32, shape=[96, 96, 3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 调用saver.restore()函数，加载训练好的网络模型
                # print('Loading success')
            else:
                print('No checkpoint')
            prediction = sess.run(logits, feed_dict={x: image})
            return prediction


txt_name = 'train.txt'
IMG_W = 96
IMG_H = 96
BATCH_SIZE = 200
CAPACITY = 64
MAX_STEP = 1000000
learning_rate = 0.00001
N_CLASSES = 56
# run_training(txt_name)
#
# # for i in range(100):
# #     i=i+13080
# #     img_path='./image_face/sface_'+str(i)+'.jpg'


file_path = './image_test'
# file_path ='E:/face68/trainb'
for file in os.listdir(file_path):
    img_path = file_path + '/' + file

    # # for i in range(9):
    # #     i = i + 1
    # #     img_path='../image/gray_PP0'+str(i)+'.jpg'
    img = cv.imread(img_path)

    prediction = val(img_path)

    img = cv.resize(img, (480, 480), interpolation=cv.INTER_CUBIC)
    for i in range(int(len(prediction[0]) / 2)):
        cv.circle(img, (int(prediction[0][i * 2] * img.shape[1]), int(prediction[0][i * 2 + 1] * img.shape[0])), 2,
                  (0, 255, 255), -1)

    # e1 = prediction[0][0:2]
    # e2 = prediction[0][2:4]
    # n1 = prediction[0][4:6]
    # m1 = prediction[0][6:8]
    # m2 = prediction[0][8:10]
    # # print('e1,e2, n1,m1,m2',e1,e2, n1,m1,m2)
    # cv.circle(img, (int(e1[0] * img.shape[1]), int(e1[1] * img.shape[0])), 2, (0, 255, 255), -1)
    # cv.circle(img, (int(e2[0] * img.shape[1]), int(e2[1] * img.shape[0])), 2, (0, 255, 255), -1)
    # cv.circle(img, (int(n1[0] * img.shape[1]), int(n1[1] * img.shape[0])), 2, (255, 0, 255), -1)
    # cv.circle(img, (int(m1[0] * img.shape[1]), int(m1[1] * img.shape[0])), 2, (255, 255, 0), -1)
    # cv.circle(img, (int(m2[0] * img.shape[1]), int(m2[1] * img.shape[0])), 2, (255, 255, 0), -1)
    cv.imshow('img', img)
    cv.waitKey()
    cv.destroyAllWindows()
    # print('预测位置',prediction)