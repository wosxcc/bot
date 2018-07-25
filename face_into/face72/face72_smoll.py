import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.python.framework import graph_util

label_lines = []
image_lines = []


def read_txt(txt_name):
    txt_open = open(txt_name)
    txt_read = txt_open.read()
    txt_lines = txt_read.split('\n')

    for line in txt_lines:
        xlabel = []
        if len(line)>3:
            line_list = line.split(' ')
            image_lines.append(line_list[0])
            xlabel.append(line_list[1])
            xlabel.append(line_list[2])
            for x in range(14):
                xlabel.append(line_list[117 + 2 + x * 2])
                xlabel.append(line_list[117 + 2 + x * 2 + 1])
            label_lines.append(xlabel)

    label_linesc=[[float(i) for i in xline] for xline in label_lines]

    return image_lines,label_linesc


def val_read(txt_name):
    image_lines, label_lines =read_txt(txt_name)
    for i in range(len(image_lines)):
        img=cv.imread(image_lines[i])
        labelss= [float(x) for  x in label_lines[i]]
        for x in range(14):
            img = cv.circle(img, (int(labelss[2 + x * 2] * img.shape[1]), int(labelss[2 + x * 2 + 1] * img.shape[0])), 1, (0, 255, 0), -1)
        cv.imshow('img',img)
        cv.waitKey()

# val_read('trainc.txt')
def get_batch(image_lines, label_lines,img_W,img_H,batch_size,capacity):
    image = tf.cast(image_lines,tf.string)

    input_queue = tf.train.slice_input_producer([image,label_lines])
    label_lines= input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image =tf.image.decode_jpeg(image_contents,channels=3)

    image =tf.image.resize_image_with_crop_or_pad(image,img_W,img_H)
    image = tf.image.random_brightness(image, max_delta=0.5)  ##在-0.5到0.5之间随机调整亮度
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  ###在-0.5到0.5之间随机调整亮度
    image = tf.image.random_hue(image, 0.5)  ##在0-0.5之间随机调整图像饱和度
    image = tf.image.per_image_standardization(image)
    img_batch ,lab_batch = tf.train.batch([image,label_lines],
                                          batch_size=batch_size,
                                          num_threads=32,
                                          capacity=capacity)
    lab_batch = tf.reshape(lab_batch, [batch_size,N_CLASSES])
    img_batch = tf.cast(img_batch, tf.float32)

    return img_batch,lab_batch


def face_net(images,lab_data,batch_size, n_classes):

    with tf.variable_scope('conv1') as scope:
        W1 = tf.get_variable('weights1', shape=(3, 3, 3, 32), dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b1 = tf.get_variable('biases1', shape=[32], dtype=tf.float32
                             , initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, W1, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, b1)
        relu1 = tf.nn.relu(pre_activation, name="relu1")


    with tf.variable_scope('conv2') as scope:
        W2 = tf.get_variable('weights2', shape=(3, 3, 32, 64), dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b2 = tf.get_variable('biases2', shape=[64], dtype=tf.float32
                             , initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(relu1, W2, strides=[1, 2, 2, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name='relu2')


    with tf.variable_scope('conv3') as scope:
        W3 = tf.get_variable('weights3', shape=(3, 3, 64, 128), dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b3 = tf.get_variable('biases3', shape=[128], dtype=tf.float32
                             , initializer=tf.constant_initializer(0.1))
        conv3 = tf.nn.conv2d(relu2, W3, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name='relu3')


    with tf.variable_scope('conv4') as scope:
        W4 = tf.get_variable('weights4', shape=(3, 3, 128, 256), dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b4 = tf.get_variable('biases4', shape=[256], dtype=tf.float32
                             , initializer=tf.constant_initializer(0.1))
        conv4 = tf.nn.conv2d(relu3, W4, strides=[1, 2, 2, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, b4), name='relu4')

    #
    # with tf.variable_scope('conv5') as scope:
    #     W5 = tf.get_variable('weights5', shape=(3, 3, 256, 512), dtype=tf.float32,
    #                          initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
    #     b5 = tf.get_variable('biases5', shape=[512], dtype=tf.float32
    #                          , initializer=tf.constant_initializer(0.1))
    #     conv5 = tf.nn.conv2d(relu4, W5, strides=[1, 1, 1, 1], padding='SAME')
    #     relu5 = tf.nn.relu(tf.nn.bias_add(conv5, b5), name='relu5')
    #
    #
    # with tf.variable_scope('conv6') as scope:
    #     W6 = tf.get_variable('weights6', shape=(3, 3, 512, 256), dtype=tf.float32,
    #                          initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    #     b6 = tf.get_variable('biases6', shape=[256], dtype=tf.float32
    #                          , initializer=tf.constant_initializer(0.1))
    #     conv6 = tf.nn.conv2d(relu5, W6, strides=[1, 2, 2, 1], padding='SAME')
    #     relu6 = tf.nn.relu(tf.nn.bias_add(conv6, b6), name='relu6')


    with tf.variable_scope('conv7') as scope:
        W7 = tf.get_variable('weights7', shape=(3, 3, 256, 128), dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        b7= tf.get_variable('biases7', shape=[128], dtype=tf.float32
                             , initializer=tf.constant_initializer(0.1))
        conv7 = tf.nn.conv2d(relu4, W7, strides=[1, 2, 2, 1], padding='SAME')
        relu7 = tf.nn.relu(tf.nn.bias_add(conv7, b7), name='relu7')

        # 全连接层
    with tf.variable_scope("fc1") as scope:
        reshape = tf.reshape(relu7, shape=[batch_size, -1])
        # print('reshape',reshape)
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[256, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        y_conv = tf.add(tf.matmul(fc1, weights), biases, name="softmax_linear")
        rmse = tf.sqrt(tf.reduce_mean(tf.square(lab_data - y_conv)))
    return y_conv, rmse

def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op



def run_training(txt_name):
    logs_train_dir = './face72/smoll/'
    train, train_into = read_txt(txt_name)
    train_batch, train_into_batch = get_batch(train, train_into,
                                              IMG_W,
                                              IMG_H,
                                              BATCH_SIZE,
                                              CAPACITY)

    train_logits, rmse = face_net(train_batch, train_into_batch, BATCH_SIZE, N_CLASSES)
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
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['softmax_linear/softmax_linear'])
                print('Step %d,train loss = %.5f' % (step, tra_loss))
                with tf.gfile.FastGFile(logs_train_dir + 'model1.pb', mode='wb') as f:
                    f.write(constant_graph.SerializeToString())


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



txt_name= 'trains.txt'
IMG_W = 96
IMG_H = 96

BATCH_SIZE = 64
CAPACITY = 64
MAX_STEP = 1000000
learning_rate = 0.00001
N_CLASSES = 30
run_training(txt_name)



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
    log_dir = 'E:/xbot/face_into/face68/image_test'
    # image_arr=test_file
    image_arr = get_one_image(test_file)
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)  ###归一化操作
        image = tf.reshape(image, [1, 96, 96, 3])
        op_intp = np.zeros(N_CLASSES, np.float32)
        p, r = face_net(image, op_intp, 1, N_CLASSES)
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
                print('没有保存的模型')
            prediction = sess.run(logits, feed_dict={x: image_arr})
            return prediction

file_path = '../face68/image_test'
# file_path ='E:/face68/trainb'
# file_path ='E:/face into'
# file_path ='E:/face72/trainb'
# file_path ='E:/face68/trainb'
for file in os.listdir(file_path):
    img_path = file_path + '/' + file
    img = cv.imread(img_path)
    start_time = datetime.datetime.now()
    prediction = val(img_path)
    print('耗时：',datetime.datetime.now()-start_time     )
    img = cv.resize(img, (480, 480), interpolation=cv.INTER_CUBIC)
    print( prediction[0][0:2])
    biaoq ='None'
    if prediction[0][0]>= 0.8 and prediction[0][0]<1.6:
        biaoq = 'Smile'
    elif prediction[0][0]>=1.6:
        biaoq = 'Laugh'
    biaoq+=':' + str(prediction[0][1])
    img = cv.putText(img, biaoq, (0, 30), 2, cv.FONT_HERSHEY_PLAIN, (255, 0, 0))
    for i in range(int(len(prediction[0]) / 2)-1):
        cv.circle(img, (int(prediction[0][2+i * 2] * img.shape[1]), int(prediction[0][2+i * 2 + 1] * img.shape[0])), 2,
                  (0, 255, 255), -1)

    cv.imshow('img', img)
    cv.waitKey()
    cv.destroyAllWindows()