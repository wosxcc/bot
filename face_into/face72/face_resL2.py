import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.python.framework import graph_util
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


def face_net(batch_size,height, width, n_classes,learning_rate):
    print(batch_size,height, width, n_classes,learning_rate)
    x = tf.placeholder(tf.float32, shape=[None, height, width, 3], name='input')
    y = tf.placeholder(tf.float32, shape=[None, n_classes], name='labels')

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


    # with tf.variable_scope('conv6') as scope:
    #     W6 = weight_variable([3, 3, 512, 256])
    #     b6 = bias_variable([256])
    #     conv6 = tf.nn.conv2d(relu5, W6, strides=[1, 2, 2, 1], padding='SAME')
    #     relu6 = tf.nn.relu(tf.nn.bias_add(conv6, b6), name='relu6')

    print(relu3)

    with tf.variable_scope('conv33') as scope:
        W33 = weight_variable([1, 1, 128, 128])
        b33= bias_variable([128])
        conv33 = tf.nn.conv2d(relu3, W33, strides=[1, 2, 2, 1], padding='SAME')
        relu33 = tf.nn.relu(tf.nn.bias_add(conv33, b33), name='relu33')

    # print(relu33)
    print(relu5)
    # resized5 = tf.image.resize_images(relu5, [48, 48], method=0)
    # print(resized5)

    # kernel = tf.constant(1.0, shape=[3, 3, 3, 128])
    # print(tf.nn.conv2d_transpose(relu5,kernel,output_shape=[-1, 48, 48, 128],strides=[1,2,2,1], padding='SAME'))

    relu66 = relu33  +relu5     ##           tf.nn.relu(tf.concat([relu3, resized5],3,name='concat1'), name='relu66')
    # concat1=tf.concat([relu3, resized5],3,name='concat1')
    # print(concat1)
    print('看看是什么',relu66)
    with tf.variable_scope('conv7') as scope:
        W7 = weight_variable([3, 3, 128, 128])
        b7= bias_variable([128])
        conv7 = tf.nn.conv2d(relu66, W7, strides=[1, 2, 2, 1], padding='SAME')
        relu7 = tf.nn.relu(tf.nn.bias_add(conv7, b7), name='relu7')


        # 全连接层
    with tf.variable_scope("fc1") as scope:

        dim = int(np.prod(relu7.get_shape()[1:]))
        reshape = tf.reshape(relu7, [-1, dim])
        weights1 =weight_variable([dim, 300])
        biases1 = bias_variable([300])
        fc1 = tf.nn.relu(tf.matmul(reshape, weights1) + biases1, name="fc1")

    with tf.variable_scope("output") as scope:
        weights2 = weight_variable([300, n_classes])
        biases2 = bias_variable([n_classes])
        y_conv = tf.add(tf.matmul(fc1, weights2), biases2, name="output")
    rmse = tf.sqrt(tf.reduce_mean(tf.square( y - y_conv)))

    # with tf.variable_scope('costs'):
    #     cost =tf.reduce_mean(rmse,name='rmse')
    #     costs = []
    #     for var in tf.trainable_variables():
    #         if var.op.name.find(r'DW')>0:
    #             costs.append(tf.nn.l2_loss(var))
    #     if len(costs)>0:
    #         cost +=tf.multiply(0.002,tf.add_n(costs))

    with tf.name_scope("optimizer"):
        optimize = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimize.minimize(rmse, global_step=global_step)
    return dict(
        x=x,
        y=y,
        optimize=train_op,
        cost=rmse,
    )



def run_training(txt_name):
    logs_train_dir = './face72/facel2/'
    X_data, Y_data = read_img(txt_name)
    graph= face_net(BATCH_SIZE, IMG_H,IMG_W, N_CLASSES,learning_rate)
    # summary_op = tf.summary.merge_all()
    sess = tf.Session()
    # train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    y_step = 0
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        # print(global_step.split('-')(1))
        y_step = int(float(global_step.split('-')[0]))
    for step in np.arange(MAX_STEP):
        for i in range(BATCH_SIZE):
            xb= (step%664)*16+i
            # ximage=np.array(X_data[xb]*255, dtype='uint8')
            # for xxi in range(72):
            #     cv.circle(ximage,(int(Y_data[xb][2+2*xxi]*96),int(Y_data[xb][2+2*xxi+1]*96)),2,(0, 255, 255), -1)
            # cv.imshow('ximage',ximage)
            # cv.waitKey()

            _ ,tra_loss= sess.run([graph['optimize'],graph['cost']],feed_dict={
                        graph['x']: np.reshape(X_data[xb], (1, 96, 96, 3)),
                        graph['y']: np.reshape(Y_data[xb], (1, 30))})

             # = sess.run(, feed_dict={
             #    graph['x']: np.reshape(X_data[xb], (1, 96, 96, 3)),
             #    graph['y']: np.reshape(Y_data[xb], (1, 30))})


        if step % 50 == 0:
            print('Step %d,train loss = %.5f' % (step+y_step, tra_loss))
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                       ['output/output'])
            with tf.gfile.FastGFile(logs_train_dir + 'face72.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())


            # 每迭代50次，打印出一次结果
            # summary_str = sess.run(summary_op)
            # train_writer.add_summary(summary_str, step)
        if step % 200 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step+y_step)
            # 每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用
    sess.close()



txt_name= 'trains.txt'
IMG_W = 96
IMG_H = 96

BATCH_SIZE = 16
CAPACITY = 16
MAX_STEP = 5000
learning_rate = 0.00001
N_CLASSES = 30
run_training(txt_name)


#
# def get_one_image(img_dir):
#     image = cv.imread(img_dir)
#     # 好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
#     bei_x = IMG_W / int(image.shape[1])
#     bei_y = IMG_H / int(image.shape[0])
#     min_bian = min(image.shape[0], image.shape[1])
#     max_bian = max(image.shape[0], image.shape[1])
#     image = cv.resize(image, None, fx=bei_x, fy=bei_y, interpolation=cv.INTER_CUBIC)
#     image_arr = np.array(image)
#
#     return image_arr
#
#
# def val(test_file):
#     log_dir = './face72/facepbx/'
#     # image_arr=test_file
#     image_arr = get_one_image(test_file)
#     with tf.Graph().as_default():
#         image = tf.cast(image_arr, tf.float32)
#         image = tf.image.per_image_standardization(image)  ###归一化操作
#         image = tf.reshape(image, [1, IMG_W, IMG_H, 3])
#         op_intp = np.zeros(N_CLASSES, np.float32)
#         p, r = face_net(image, op_intp, 1, N_CLASSES)
#         # print('看看p的值：',p)
#         logits = p  # tf.nn.softmax(p)
#         x = tf.placeholder(tf.float32, shape=[IMG_W, IMG_H, 3])
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