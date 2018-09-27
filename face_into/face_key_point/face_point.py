import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import datetime
import random
from tensorflow.python.ops import control_flow_ops
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
            img = cv.imread(line_list[0])
            img = (np.array(img,dtype='float32')-127.5) /128
            image_lines.append(img)
            a_label = [float(i) for i in line_list[1:]]
            a_label[0] = a_label[0]/2
            a_label = np.array(a_label,dtype='float32')
            a_label = (a_label-0.5)*2.0
            # print(a_label)
            label_lines.append(a_label)

    label_linesc=[[float(i) for i in xline] for xline in label_lines]
    ximage_lines=np.array(image_lines, dtype='float32')

    xlabel_linesc=np.array(label_linesc, dtype='float32')
    return ximage_lines,xlabel_linesc



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

    ss= '%.5f'%(y)
    if len(ss)>=7:
        ss = ss[0:7]
    else:
        for i in range(7-len(ss)):
            ss+= '0'
    put_str='step:%d  loss:'%(x)+ss
    img[120:180,500:920,:]=255
    cv.putText(img, put_str,(500,150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    spot_x = max(min(int(x/MAX_STEP*1000+100),1000),0)
    spot_y = max(min(int(900-y*1000),1000),0)
    # print('画点位置：',spot_x,spot_y)
    cv.circle(img,(spot_x,spot_y),3,(0,0,255),-1)
    cv.imshow('LOSS',img)
    cv.waitKey(10)


def face_net(batch_size,height, width, n_classes,learning_rate,phase_train):
    print(batch_size,height, width, n_classes,learning_rate)
    x = tf.placeholder(tf.float32, shape=[None, height, width, 3], name='input')
    y = tf.placeholder(tf.float32, shape=[None, n_classes], name='labels')

    def batch_norm(x, phase_train):  # pylint: disable=unused-variable
        name = 'batch_norm'
        with tf.variable_scope(name):
            phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
            n_out = int(x.get_shape()[-1])
            beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                               name=name + '/beta', trainable=True, dtype=x.dtype)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                                name=name + '/gamma', trainable=True, dtype=x.dtype)

            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = control_flow_ops.cond(phase_train,
                                              mean_var_with_update,
                                              lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def weight_variable(shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    with tf.variable_scope('conv1') as scope:
        W1 = weight_variable([7, 7, 3, 32])
        b1 = bias_variable([32])
        conv = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, b1)
        relu1 =  batch_norm(pre_activation, phase_train)

    with tf.variable_scope('conv2') as scope:
        W2 = weight_variable([5, 5, 32, 64])
        b2 = bias_variable([64])
        conv2 = tf.nn.conv2d(relu1, W2, strides=[1, 2, 2, 1], padding='SAME')
        relu2 =  batch_norm(tf.nn.bias_add(conv2, b2), phase_train)



    with tf.variable_scope('conv3') as scope:
        W3 = weight_variable([5, 5, 64, 128])
        b3 = bias_variable([128])
        conv3 = tf.nn.conv2d(relu2, W3, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = batch_norm(tf.nn.bias_add(conv3, b3), phase_train)

    with tf.variable_scope('conv4') as scope:
        W4 = weight_variable([3, 3, 128, 256])
        b4 = bias_variable([256])
        conv4 = tf.nn.conv2d(relu3, W4, strides=[1, 2, 2, 1], padding='SAME')
        relu4 = batch_norm(tf.nn.bias_add(conv4, b4), phase_train)


    with tf.variable_scope('conv5') as scope:
        W5 = weight_variable([3, 3, 256, 256])
        b5 = bias_variable([256])
        conv5 = tf.nn.conv2d(relu4, W5, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = batch_norm(tf.nn.bias_add(conv5, b5), phase_train)


    with tf.variable_scope('conv6') as scope:
        W6 = weight_variable([3, 3, 256, 512])
        b6 = bias_variable([512])
        conv6 = tf.nn.conv2d(relu5, W6, strides=[1, 2, 2, 1], padding='SAME')
        relu6 = tf.nn.relu(tf.nn.bias_add(conv6, b6), name='relu6')

    with tf.variable_scope('conv8') as scope:
        W8 = weight_variable([3, 3, 512, 256])
        b8 = bias_variable([256])
        conv8 = tf.nn.conv2d(relu6, W8, strides=[1, 1, 1, 1], padding='SAME')
        relu8 = tf.nn.relu(tf.nn.bias_add(conv8, b8), name='relu8')

    with tf.variable_scope('conv7') as scope:
        W7 = weight_variable([3, 3, 256, 128])
        b7= bias_variable([128])
        conv7 = tf.nn.conv2d(relu8, W7, strides=[1, 1, 1, 1], padding='SAME')
        relu7 = batch_norm(tf.nn.bias_add(conv7, b7),phase_train)



        # 全连接层
    with tf.variable_scope("fc1") as scope:

        dim = int(np.prod(relu7.get_shape()[1:]))
        reshape = tf.reshape(relu7, [-1, dim])
        weights1 =weight_variable([dim, 256])   ##24*24*256*256
        biases1 = bias_variable([256])
        fc1 = batch_norm(tf.matmul(reshape, weights1) + biases1,phase_train)

    with tf.variable_scope("fc2") as scope:
        weights122 =weight_variable([256, 256])
        biases122 = bias_variable([256])
        fc2 = batch_norm(tf.matmul(fc1, weights122) + biases122, phase_train)

    with tf.variable_scope("output") as scope:
        weights2 = weight_variable([256, n_classes])
        biases2 = bias_variable([n_classes])
        y_conv=tf.add(tf.matmul(fc2, weights2),biases2,name="output")
        rmse = tf.reduce_sum(tf.square(y - y_conv),name = 'loss')

    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(rmse, global_step=global_step)

    return dict(
        x=x,
        y=y,
        y_conv=y_conv,
        optimize=train_op,
        cost=rmse,
    )


def run_training(txt_name):
    imgs = draw_form(MAX_STEP)
    logs_train_dir = './face_key/0925/'
    X_data, Y_data = read_img(txt_name)
    graph= face_net(BATCH_SIZE, IMG_H,IMG_W, N_CLASSES,learning_rate,True)
    # summary_op = tf.summary.merge_all()
    sess = tf.Session()
    # train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
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
    loss_list['x']= []
    loss_list['y'] = []

    # max_batch =X_data.shape[0]//BATCH_SIZE
    #
    # print('max_batch',max_batch)
    for step in np.arange(MAX_STEP):


        # img_m = np.zeros((960,960,3),np.uint8)
        #
        # for xb in  range(BATCH_SIZE):
        #     imgc = X_data[(step % max_batch) * BATCH_SIZE+xb]*128 +127.5
        #     imga = np.zeros((imgc.shape),np.uint8)
        #     imga[:,:,:] =imgc
        #     for xxx in range(int((N_CLASSES-2)/2)):
        #        nx =int((Y_data[(step % max_batch) * BATCH_SIZE+xb][xxx*2+2]+1)/2 *IMG_W)
        #        ny =int((Y_data[(step % max_batch) * BATCH_SIZE+xb][xxx*2+3]+1)/2*IMG_W)
        #        cv.circle(imga,(nx,ny),2,(0,255,255),-1)
        #     imga=cv.resize(imga,(120,120),interpolation=cv.INTER_CUBIC)
        #     # cv.imshow('imga', imga)
        #     # cv.waitKey()
        #     img_m[xb//8*120:xb//8*120+120,xb%8*120:xb%8*120+120,:] = imga
        #
        # cv.imshow('img_m',img_m)
        # cv.waitKey()


        # (step % max_batch) * BATCH_SIZE: (step % max_batch) * BATCH_SIZE + BATCH_SIZE
        # (step%max_batch)*BATCH_SIZE:(step%max_batch)*BATCH_SIZE+BATCH_SIZE


        batch_img = []
        batch_lab = []
        for ai in range(BATCH_SIZE):
            xxx = random.randint(0,X_data.shape[0]-1)
            batch_img.append(X_data[xxx])
            batch_lab.append(Y_data[xxx])
        _, tra_loss , tra_y_conv ,input_y= sess.run([graph['optimize'],graph['cost'],graph['y_conv'],graph['y']],feed_dict={
                    graph['x']: np.reshape(batch_img, (BATCH_SIZE, IMG_H, IMG_W, 3)),
                    graph['y']: np.reshape(batch_lab, (BATCH_SIZE, N_CLASSES))})
        print('得到的值',tra_y_conv)
        print('输入的值',input_y)
        loss_list['x'].append(step+y_step)
        loss_list['y'].append(tra_loss)
        drow_spot(imgs,step, tra_loss, MAX_STEP)

        if step % 50 == 0:
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



txt_name= 'trainc.txt'
IMG_W = 160
IMG_H = 160

BATCH_SIZE = 64
MAX_STEP = 80000
learning_rate = 0.00001
N_CLASSES = 146
# run_training(txt_name)




def get_one_image(img_dir):
    image = cv.imread(img_dir)
    # 好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
    bei_x = IMG_W / int(image.shape[1])
    bei_y = IMG_H / int(image.shape[0])
    image = cv.resize(image, None, fx=bei_x, fy=bei_y, interpolation=cv.INTER_CUBIC)
    image_arr = np.array(image)

    return image_arr


def val(test_file):
    log_dir = './face_key/0925/'
    # image_arr=test_file
    image_arr = get_one_image(test_file)
    with tf.Graph().as_default():
        image =(image_arr-127.5)/128
        op_intp = np.zeros(N_CLASSES, np.float32)
        graph= face_net(1,IMG_W, IMG_H, N_CLASSES,learning_rate,False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            print('看看值',ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('没有保存的模型')
            prediction = sess.run(graph['y_conv'] , feed_dict={graph['x']: np.reshape(image, (1, IMG_W, IMG_H, 3)),graph['y']:np.reshape(op_intp, (1, N_CLASSES))})
            return prediction

file_path = '../face68/image_test'
for file in os.listdir(file_path):
    img_path = file_path + '/' + file
    img = cv.imread(img_path)
    start_time = datetime.datetime.now()
    prediction = val(img_path)
    print(prediction)
    print('耗时：',datetime.datetime.now()-start_time)
    img = cv.resize(img, (480, 480), interpolation=cv.INTER_CUBIC)
    print( prediction[0][0:2])

    biaoq ='None'

    prediction[0]= (prediction[0]+1)/2
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