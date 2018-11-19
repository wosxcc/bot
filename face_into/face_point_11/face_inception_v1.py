import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import datetime
import random
import cv2 as cv
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import graph_util
import os
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


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net


def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                     tower_conv2_2, tower_pool], 3)
    return net


def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
                                   dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size,
                                   reuse=reuse)


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                                  scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net

                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)
                end_points['Mixed_5a'] = net

                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net

                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)
                end_points['Mixed_6b'] = net

                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net

                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                end_points['Mixed_8a'] = net

                net = block8(net, activation_fn=None)
                end_points['Mixed_8b'] = net

                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    # pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)

                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')

                    end_points['PreLogitsFlatten'] = net

                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                                           scope='Bottleneck', reuse=False)
    net =tf.add(net,0,name= 'output')
    return net, end_points

def retu_loss(y_data,y_count):
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_data - y_count), name='loss')
    return loss

def optim_opp(loss,learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return  train_op
def run_training(txt_name):
    imgs = draw_form(MAX_STEP)
    read_train_dir = './face_bot_v1/11_18/'
    seave_train_dir = './face_bot_v1/11_18/'
    X_data, Y_data = read_img(txt_name)
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_H, 3], name='input')
    y = tf.placeholder(tf.float32, shape=[None, N_CLASSES], name='labels')

    y_count,s_= inference(x, 0.8, phase_train,
              bottleneck_layer_size=146, weight_decay=0.00, reuse=None)
    floss =retu_loss(y_count,y)

    train_op = optim_opp(floss,learning_rate)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(read_train_dir)
    y_step=0
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('原有训练次数',global_step)
        y_step = int(float(global_step))

    loss_list ={}
    loss_list['x']= []
    loss_list['y'] = []

    # max_batch =X_data.shape[0]//BATCH_SIZE
    #
    # print('max_batch',max_batch)
    batch_img = []
    batch_lab = []
    for ai in range(BATCH_SIZE):
        xxx = random.randint(0, X_data.shape[0] - 1)
        batch_img.append(X_data[xxx])
        batch_lab.append(Y_data[xxx])
    for step in np.arange(MAX_STEP):

        # batch_img = []
        # batch_lab = []
        # for ai in range(BATCH_SIZE):
        #     xxx = random.randint(0,X_data.shape[0]-1)
        #     batch_img.append(X_data[xxx])
        #     batch_lab.append(Y_data[xxx])
        _, tra_loss , tra_y_conv ,ss_= sess.run([train_op,floss,y_count,s_],feed_dict={
                    x: np.reshape(batch_img, (BATCH_SIZE, IMG_H, IMG_W, 3)),
                    y: np.reshape(batch_lab, (BATCH_SIZE, N_CLASSES)),
                    phase_train:True})
        loss_list['x'].append(step+y_step)
        loss_list['y'].append(tra_loss)
        drow_spot(imgs,step, tra_loss, MAX_STEP)

        if step % 50 == 0:
            print('Step %d,train loss = %.5f' % (step+y_step, tra_loss))
            gd = sess.graph.as_graph_def()

                # fix batch norm nodes
            for node in gd.node:
                # print(node.name)
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                       ['output']) # InceptionResnetV2/Bottleneck/BatchNorm/Reshape_1

            with tf.gfile.FastGFile(seave_train_dir + 'face72.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            # 每迭代50次，打印出一次结果
        if step % 200 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(seave_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step+y_step)
            # 每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用
    sess.close()




txt_name= 'trainc.txt'
IMG_W = 160
IMG_H = 160

BATCH_SIZE = 50
MAX_STEP = 2000
learning_rate = 0.005
N_CLASSES = 146
run_training(txt_name)



def get_one_image(img_dir):
    image = cv.imread(img_dir)
    # 好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
    bei_x = IMG_W / int(image.shape[1])
    bei_y = IMG_H / int(image.shape[0])
    image = cv.resize(image, None, fx=bei_x, fy=bei_y, interpolation=cv.INTER_CUBIC)
    image_arr = np.array(image)

    return image_arr

file_path = 'E:/face72/trainb'
log_dir = './face_bot_v1/11_18/'
# image_arr=test_file
with tf.Graph().as_default():
    op_intp = np.zeros(N_CLASSES, np.float32)
    #graph= face_net(1,IMG_W, IMG_H, N_CLASSES,learning_rate,False)

    phase_train = tf.placeholder(tf.bool, name='phase_train')
    x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_H, 3], name='input')
    y_count, s_ = inference(x, 0.8, phase_train,
                            bottleneck_layer_size=146, weight_decay=0.001, reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(log_dir)
        print('看看值',ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            for file in os.listdir(file_path):
                img_path = file_path + '/' + file
                img = cv.imread(img_path)
                start_time = datetime.datetime.now()
                image_arr = get_one_image(img_path)
                image = (image_arr - 127.5) / 128
                prediction = sess.run(y_count, feed_dict={x: np.reshape(image, (1, IMG_W, IMG_H, 3)),
                                                                  phase_train:False})
                print('耗时：',datetime.datetime.now()-start_time)
                img = cv.resize(img, (480, 480), interpolation=cv.INTER_CUBIC)
                biaoq ='None'
                prediction[0]= (prediction[0]+1)/2
                if prediction[0][0]>= 0.4 and prediction[0][0]<0.8:
                    biaoq = 'Smile'
                elif prediction[0][0]>=0.8:
                    biaoq = 'Laugh'
                biaoq+=':' + str(prediction[0][1])
                img = cv.putText(img, biaoq, (0, 30), 2, cv.FONT_HERSHEY_PLAIN, (255, 0, 0))
                for i in range(int(len(prediction[0]) / 2)-1):
                    cv.circle(img, (int(prediction[0][2+i * 2] * img.shape[1]), int(prediction[0][2+i * 2 + 1] * img.shape[0])), 2,
                              (0, 255, 255), -1)
                cv.imshow('img', img)
                cv.waitKey()
                cv.destroyAllWindows()
        else:
            print('没有保存的模型')