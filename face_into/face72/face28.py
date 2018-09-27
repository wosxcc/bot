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




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_Variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return initial
def conv2d(x,W,C):
    return tf.nn.conv2d(x,W,strides=C,padding="SAME")
def padding_2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")





def run_training(txt_name,batch_size,n_classes):

    C1 =[1,1,1,1]
    C2= [1,2,2,1]
    logs_train_dir = './face72/smoll/pb/'
    train, train_into = read_txt(txt_name)
    train_batch, train_into_batch = get_batch(train, train_into,
                                              IMG_W,
                                              IMG_H,
                                              BATCH_SIZE,
                                              CAPACITY)

    x = tf.placeholder("float32", [None, 27648], name='input_x')
    y_ = tf.placeholder("float32", [None, n_classes], name='input_y')

    w_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_Variable([32])
    x_image = tf.reshape(x, [-1, 96, 96, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1,C1) + b_conv1)

    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_Variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2,C2) + b_conv2)


    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_Variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name="fc1")

    keep_prob = tf.placeholder("float32", name='keep_prob')

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_Variable([10])



    W1 = tf.get_variable('weights1', shape=(3, 3, 3, 32), dtype=tf.float32
                         ,initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b1 = tf.get_variable('biases1', shape=[32], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(train_batch, W1, strides=[1, 1, 1, 1], padding="SAME")
    pre_activation = tf.nn.bias_add(conv, b1)
    relu1 = tf.nn.relu(pre_activation, name="relu1")


    W2 = tf.get_variable('weights2', shape=(3, 3, 32, 64), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b2 = tf.get_variable('biases2', shape=[64], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv2 = tf.nn.conv2d(relu1, W2, strides=[1, 2, 2, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name='relu2')


    W3 = tf.get_variable('weights3', shape=(3, 3, 64, 128), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b3 = tf.get_variable('biases3', shape=[128], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv3 = tf.nn.conv2d(relu2, W3, strides=[1, 1, 1, 1], padding='SAME')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name='relu3')


    W4 = tf.get_variable('weights4', shape=(3, 3, 128, 256), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b4 = tf.get_variable('biases4', shape=[256], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv4 = tf.nn.conv2d(relu3, W4, strides=[1, 2, 2, 1], padding='SAME')
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, b4), name='relu4')

    W7 = tf.get_variable('weights7', shape=(3, 3, 256, 128), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b7= tf.get_variable('biases7', shape=[128], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv7 = tf.nn.conv2d(relu4, W7, strides=[1, 2, 2, 1], padding='SAME')
    relu7 = tf.nn.relu(tf.nn.bias_add(conv7, b7), name='relu7')

    reshape = tf.reshape(relu7, shape=[batch_size, -1])
    # print('reshape',reshape)
    dim = reshape.get_shape()[1].value
    weights8 = tf.get_variable("weights",
                              shape=[dim, 256],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    biases8 = tf.get_variable("biases",
                             shape=[256],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshape, weights8) + biases8, name="fc1")

    weights9 = tf.get_variable("weights",
                              shape=[256, n_classes],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    biases9 = tf.get_variable("biases",
                             shape=[n_classes],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    y_conv = tf.add(tf.matmul(fc1, weights9), biases9, name="softmax_linear")
    rmse = tf.sqrt(tf.reduce_mean(tf.square(train_into_batch - y_conv)))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = optimizer.minimize(rmse, global_step=global_step)


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
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['softmax_linear'])
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
# run_training(txt_name,BATCH_SIZE,N_CLASSES)



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


# def val(test_file):
#     log_dir = './face72/smoll/'
#     # image_arr=test_file
#     image_arr = get_one_image(test_file)
#     with tf.Graph().as_default():
#         image = tf.cast(image_arr, tf.float32)
#         image = tf.image.per_image_standardization(image)  ###归一化操作
#         image = tf.reshape(image, [1, 96, 96, 3])
#         op_intp = np.zeros(N_CLASSES, np.float32)
#         p, r = face_net(image, op_intp, 1, N_CLASSES)
#         # print('看看p的值：',p)
#         logits = p  # tf.nn.softmax(p)
#         x = tf.placeholder(tf.float32, shape=[96, 96, 3])
#         saver = tf.train.Saver()
#         with tf.Session() as sess:
#             ckpt = tf.train.get_checkpoint_state(log_dir)
#             if ckpt and ckpt.model_checkpoint_path:
#                 global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 # 调用saver.restore()函数，加载训练好的网络模型
#                 # print('Loading success')
#             else:
#                 print('没有保存的模型')
#             prediction = sess.run(logits, feed_dict={x: image_arr})
#             return prediction
#
# file_path = '../face68/image_test'
# # file_path ='E:/face68/trainb'
# # file_path ='E:/face into'
# # file_path ='E:/face72/trainb'
# # file_path ='E:/face68/trainb'
# for file in os.listdir(file_path):
#     img_path = file_path + '/' + file
#     img = cv.imread(img_path)
#     start_time = datetime.datetime.now()
#     prediction = val(img_path)
#     print('耗时：',datetime.datetime.now()-start_time     )
#     img = cv.resize(img, (480, 480), interpolation=cv.INTER_CUBIC)
#     print( prediction[0][0:2])
#     biaoq ='None'
#     if prediction[0][0]>= 0.8 and prediction[0][0]<1.6:
#         biaoq = 'Smile'
#     elif prediction[0][0]>=1.6:
#         biaoq = 'Laugh'
#     biaoq+=':' + str(prediction[0][1])
#     img = cv.putText(img, biaoq, (0, 30), 2, cv.FONT_HERSHEY_PLAIN, (255, 0, 0))
#     for i in range(int(len(prediction[0]) / 2)-1):
#         cv.circle(img, (int(prediction[0][2+i * 2] * img.shape[1]), int(prediction[0][2+i * 2 + 1] * img.shape[0])), 2,
#                   (0, 255, 255), -1)
#
#     cv.imshow('img', img)
#     cv.waitKey()
#     cv.destroyAllWindows()