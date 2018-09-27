import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.python.framework import graph_util
from face_ID_net.read_image import *
from face_ID_net.ID_pb_net import face_net

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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




def run_training(txt_name):
    imgs = draw_form(MAX_STEP)
    logs_train_dir = './face72/face_big/'
    X_data = read_image()
    graph= face_net(BATCH_SIZE, IMG_H,IMG_W, N_CLASSES,learning_rate,margin=0.2,run_train=True)
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
        count_pos = 0.0
        count_neg = 0.0
        for i in range(BATCH_SIZE):
            xb= (step%82)*32+i
            _, tra_loss, sess_pos, sess_neg = sess.run([graph['optimize'],graph['loss'],graph['d_pos'],graph['d_neg']],feed_dict={
                        graph['x']: np.reshape(X_data[xb], (3, 64, 64, 3))})
            count_pos+=sess_pos
            count_neg+=sess_neg
            loss_avg+=tra_loss
        avg_loss =loss_avg/BATCH_SIZE
        loss_list['x'].append(step+y_step)
        loss_list['y'].append(avg_loss)
        # loss_list['x'].append(step+y_step)
        # loss_list['y'].append(tra_loss)
        drow_spot(imgs,step, avg_loss, MAX_STEP)
        print('同:',count_pos/32.0,'不同:',count_neg/32.0,'距离差',(count_neg-count_pos)/32)
        if step % 50 == 0:
            # print('同一个人',sess.run(tf.reduce_mean(sess_pos)),'\t',sess_pos)
            # print('不同一个人',sess.run(tf.reduce_mean(sess_neg)),'\t',sess_neg)
            print('Step %d,train loss = %.5f' % (step+y_step, avg_loss))
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
learning_rate = 0.000001
N_CLASSES = 128
run_training(txt_name)



