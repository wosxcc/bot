import os
import io
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
from image_read import readimg

NOW_CLASS='face'  ## blouse  dress   outwear   skirt   trousers
position=[]
image_payh=[]
txt_name='train_yes.txt'


def read_text(txt_name):
    file_txt = open(txt_name)
    get = file_txt.read()
    result = get.split('\n')
    other_result = get.splitlines()
    try:
        for i in range(len(other_result)):
            data_file=result[i].split('\t')
            image_payh.append(data_file[0])
            position.append(data_file[2:])
    except:
        print('Erro')
    print(len(image_payh))
    positionc = [[float(i) for i in line] for line in position]  ##字符串转数字
    # for j in range(20):
    #     print('./img_H/'+image_payh[j])
    #     img=cv.imread('./img_H/'+image_payh[j])
    #     cv.imshow('imgx', img)
    #     cv.waitKey()
    #     print('positionc',positionc[j])
    #     imgx = cv.rectangle(img, (int(positionc[j][0] * img.shape[1]), int(positionc[j][1] * img.shape[0])),
    #                         (int(positionc[j][2] * img.shape[1]), int(positionc[j][3] * img.shape[0])), (0, 0, 255), 1)
    #     cv.imshow('imgx', imgx)
    #     cv.waitKey()
    return image_payh,positionc
# read_text(txt_name)


def get_batch(image, position, img_W, img_H, batch_size, N_CLASSES, capacity):
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

    position_batch = tf.reshape(position_batch, [batch_size, N_CLASSES])
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


    # 全连接层
    with tf.variable_scope("fc1") as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
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


    # 全连接层
    with tf.variable_scope("fc3") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc3 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc3")

    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[32, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        y_conv = tf.add(tf.matmul(fc3, weights), biases, name="softmax_linear")
        rmse = tf.sqrt(tf.reduce_mean(tf.square(y_data - y_conv)))
    return y_conv, rmse


def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def run_training(txt_name):
    logs_train_dir = './face/' + NOW_CLASS + '/'
    train, train_into = read_text(txt_name)
    train_batch, train_into_batch = get_batch(train, train_into,
                                              IMG_W,
                                              IMG_H,
                                              BATCH_SIZE,
                                              N_CLASSES,
                                              CAPACITY)
    train_logits, rmse = inference(train_batch, train_into_batch, BATCH_SIZE, N_CLASSES)
    train_op = trainning(rmse, learning_rate)
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss = sess.run([train_op, rmse])
            if step % 50 == 0:
                print('第 %d次训练,损失值是 = %.5f' % (step, tra_loss))
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
    bei_x = IMG_W / int(image.shape[1])
    bei_y = IMG_H / int(image.shape[0])
    min_bian = min(image.shape[0], image.shape[1])
    max_bian = max(image.shape[0], image.shape[1])
    if img.shape[0] == min_bian:
        cha = int((img.shape[1] - min_bian) / 2)
        images = np.zeros((img.shape[1], img.shape[1], 3), np.uint8)
        images[cha:cha + min_bian, :, :] = img
        image = cv.resize(images, None, fx=bei_x, fy=bei_x, interpolation=cv.INTER_CUBIC)
    else:
        cha = int((img.shape[0] - min_bian) / 2)
        images = np.zeros((img.shape[0], img.shape[0], 3), np.uint8)
        images[:, cha:cha + min_bian, :] = img
        image = cv.resize(images, None, fx=bei_x, fy=bei_x, interpolation=cv.INTER_CUBIC)
    # image = cv.resize(image, None, fx=bei_x, fy=bei_y, interpolation=cv.INTER_CUBIC)
    image_arr = np.array(image)

    return image_arr


def test(test_file):
    log_dir = './face/' + NOW_CLASS + '/'
    # image_arr=test_file
    image_arr = get_one_image(test_file)
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, IMG_H, IMG_W, 3])
        op_intp = np.zeros(N_CLASSES, np.float32)
        p, r = inference(image, op_intp, 1, N_CLASSES)
        # print('看看p的值：',p)
        logits = p  # tf.nn.softmax(p)
        x = tf.placeholder(tf.float32, shape=[IMG_H, IMG_W, 3])
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
            prediction = sess.run(logits, feed_dict={x: image_arr})
            return prediction


txt_name = 'viedo_face.txt'
IMG_W = 48
IMG_H = 48
BATCH_SIZE = 300
CAPACITY = 64
MAX_STEP = 20000
learning_rate = 0.0001
N_CLASSES = 4
# run_training(txt_name)

pathx='./image_face_into'##+NOW_CLASS
for file in os.listdir(pathx):
    img_path = pathx+'/' + file
    img = cv.imread(img_path)
    prediction=test(img_path)
    print(prediction)

    img = cv.resize(img, None, fx=10, fy=10, interpolation=cv.INTER_CUBIC)
    cv.rectangle(img, (int(prediction[0][0] * img.shape[1]), int(prediction[0][1] * img.shape[0])), (int(prediction[0][2] * img.shape[1]), int(prediction[0][3] * img.shape[0])), (0, 255, 0),2)
        # cv.circle(img, (int(prediction[0][2*j] * img.shape[1]), int(prediction[0][2*j+1] * img.shape[0])), 2, (0, 255, 0), -1)
    cv.imshow('img', img)
    cv.waitKey()
    cv.destroyAllWindows()