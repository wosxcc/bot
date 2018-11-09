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

            img = cv.imread(line_list[0])

            img = (np.array(img,dtype='float32')-127.5) /128
            image_lines.append(img)
            # xlabel.append(line_list[1])
            # xlabel.append(line_list[2])
            # for x in range(14):
            #     xlabel.append(line_list[117 + 2 + x * 2])
            #     xlabel.append(line_list[117 + 2 + x * 2 + 1])
            # label_lines.append(xlabel)
            #
            label_lines.append(line_list[1:])


    label_linesc=[[float(i) for i in xline] for xline in label_lines]
    ximage_lines=np.array(image_lines, dtype='float32')

    xlabel_linesc=np.array(label_linesc, dtype='float32')
    return ximage_lines,xlabel_linesc


#
# def read_txt(txt_name):
#     txt_open = open(txt_name)
#     txt_read = txt_open.read()
#     txt_lines = txt_read.split('\n')
#
#     for line in txt_lines:
#         xlabel = []
#         if len(line)>3:
#             line_list = line.split(' ')
#             image_lines.append(line_list[0])
#             label_lines.append(line_list[1:])
#     label_linesc=[[float(i) for i in xline] for xline in label_lines]
#
#     return image_lines,label_linesc



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
    print(put_str)
    img[120:180,500:880,:]=255
    cv.putText(img, put_str,(500,150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    spot_x = int(x/MAX_STEP*1000+100)
    spot_y =int(900-y*1000)
    # print('画点位置：',spot_x,spot_y)
    cv.circle(img,(spot_x,spot_y),3,(0,0,255),-1)
    cv.imshow('LOSS',img)
    cv.waitKey(10)


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
        weights122 =weight_variable([256, 256])
        biases122 = bias_variable([256])
        fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc1, weights122) + biases122, name="fc2"),0.5)

    with tf.variable_scope("output") as scope:
        weights2 = weight_variable([256, n_classes])
        biases2 = bias_variable([n_classes])
        # y_conv = tf.sigmoid(tf.matmul(fc2, weights2)+biases2, name="output")
        # y_conv = tf.sigmoid(tf.matmul(fc2, weights2)+biases2, name="output")
        y_conv=tf.nn.softmax(tf.add(tf.matmul(fc2, weights2),biases2), name="output")
    yy_conv = tf.add(y_conv, 0, name='xoutput')
    # rmse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
    rmse = tf.reduce_mean(tf.square(y - y_conv))

    with tf.name_scope("optimizer"):
        optimize = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimize.minimize(rmse, global_step=global_step)
    print()
    return dict(
        x=x,
        y=y,
        weights2= [weights2,weights122],
        biases2=[biases2,biases122],
        y_conv=y_conv,
        optimize=train_op,
        cost=rmse,
    )


def run_training(txt_name):
    imgs = draw_form(MAX_STEP)
    logs_train_dir = './face_key/face_0918/'
    X_data, Y_data = read_img(txt_name)
    graph= face_net(BATCH_SIZE, IMG_H,IMG_W, N_CLASSES,learning_rate)
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
    loss_list['x']=[]
    loss_list['y'] = []

    for step in np.arange(MAX_STEP):
        # loss_avg=0.0
        # for i in range(BATCH_SIZE):
        #     xb= (step%332)*32+i
            # ximage=np.array(X_data[xb]*255+127.5, dtype='uint8')
            # for xxi in range(72):
            #     cv.circle(ximage,(int(Y_data[xb][2+2*xxi]*96),int(Y_data[xb][2+2*xxi+1]*96)),2,(0, 255, 255), -1)
            # cv.imshow('ximage',ximage)
            # cv.waitKey()
            # print(xb)
        _, tra_loss, weights2, biases2 = sess.run([graph['optimize'],graph['cost'],graph['weights2'],graph['biases2']],feed_dict={
                    graph['x']: np.reshape(X_data[(step%664)*16:(step%664)*16+16], (16, IMG_H, IMG_W, 3)),
                    graph['y']: np.reshape(Y_data[(step%664)*16:(step%664)*16+16], (16, N_CLASSES))})
            # loss_avg+=tra_loss
        # avg_loss =loss_avg/BATCH_SIZE

        loss_list['x'].append(step+y_step)
        loss_list['y'].append(tra_loss)

        drow_spot(imgs,step, tra_loss, MAX_STEP)


        # print('次数：',step,'对应loss:',tra_loss)

             # = sess.run(, feed_dict={
             #    graph['x']: np.reshape(X_data[xb], (1, 96, 96, 3)),
             #    graph['y']: np.reshape(Y_data[xb], (1, 30))})

        if step % 50 == 0:
            print('Step %d,train loss = %.5f' % (step+y_step, tra_loss))
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                       ['xoutput'])
            with tf.gfile.FastGFile(logs_train_dir + 'facexb.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())


            # 每迭代50次，打印出一次结果
            # summary_str = sess.run(summary_op)
            # train_writer.add_summary(summary_str, step)
        if step % 200 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step+y_step)
            # 每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用
    sess.close()



txt_name= 'trainc.txt'
IMG_W = 160
IMG_H = 160

BATCH_SIZE = 16
CAPACITY = 16
MAX_STEP = 6000
learning_rate = 0.0001
N_CLASSES = 146
run_training(txt_name)




def get_one_image(img_dir):
    image = cv.imread(img_dir)
    # 好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
    bei_x = IMG_W / int(image.shape[1])
    bei_y = IMG_H / int(image.shape[0])
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
    log_dir = './face_key/face_0918/'
    # image_arr=test_file
    image_arr = get_one_image(test_file)
    with tf.Graph().as_default():
        image =(image_arr-127.5)/128
        op_intp = np.zeros(N_CLASSES, np.float32)
        graph= face_net(1,IMG_W, IMG_H, N_CLASSES,learning_rate)
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
# file_path ='E:/face68/trainb'
# file_path ='E:/face into'
# file_path ='E:/face72/trainb'
# file_path ='E:/face68/trainb'
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