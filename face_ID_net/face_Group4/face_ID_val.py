import os
import cv2 as cv
import numpy as np
import  random
import tensorflow as tf
from face_ID_net.face_Group4.deep_net import face_net
# from face_ID_net.face_Group4.ID_pb_net1024 import face_net

from face_ID_net.face_Group4.read_Group4 import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
IMG_H=96
IMG_W =96
N_CLASSES =512
learning_rate =0.001

def face_val(image_arr,run_train):
    print('搞毛线啊')
    log_dir = './faceID/deep_net/'
    with tf.Graph().as_default():
        graph = face_net(1, IMG_H,IMG_W, N_CLASSES,learning_rate,2,run_train)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('没有保存的模型')
            if run_train ==True:
                pos_d,neg_d = sess.run([graph['d_pos'],graph['d_neg']],feed_dict={graph['x']: np.reshape(image_arr, (4, IMG_W, IMG_H, 3))})
                return pos_d, neg_d
            elif run_train ==False:
                print('下面出错了',len(image_arr),image_arr[0].shape)

                anchor_data = sess.run(graph['anchor_out'],feed_dict={graph['x']: np.reshape(image_arr, ( 1, IMG_W, IMG_H, 3))})
                print('上面出错了')
                return anchor_data


pacth = 'E:/faceID'
for i in range(10):
    file = random.sample(os.listdir(pacth),1)[0]
    while(1):
        negative_file= random.sample(os.listdir(pacth),1)[0]
        if negative_file!=file:
            break
    print(file,negative_file)

    anchor_img = random.sample(os.listdir(pacth+'/'+file),1)[0]
    while(1):
        positive_img = random.sample(os.listdir(pacth+'/'+file),1)[0]
        if anchor_img!=positive_img:
            break
    negative_img = random.sample(os.listdir(pacth+'/'+negative_file),1)[0]
    negative_img2 = random.sample(os.listdir(pacth + '/' + negative_file), 1)[0]

    img_anchor=cv.imread(pacth+'/'+file+'/'+anchor_img)
    img_positive=cv.imread(pacth+'/'+file+'/'+positive_img)
    img_negative=cv.imread(pacth+'/'+negative_file+'/'+negative_img)
    img_negative2 = cv.imread(pacth + '/' + negative_file + '/' + negative_img2)

    sh_anchor=cv.resize(img_anchor,(240,240),interpolation=cv.INTER_CUBIC)
    sh_positive=cv.resize(img_positive,(240,240),interpolation=cv.INTER_CUBIC)
    sh_negative=cv.resize(img_negative,(240,240),interpolation=cv.INTER_CUBIC)
    sh_negative2 = cv.resize(img_negative2, (240, 240), interpolation=cv.INTER_CUBIC)

    image_data=[]

    image_data.append(cv.resize(img_anchor,(IMG_W,IMG_H),interpolation=cv.INTER_CUBIC))
    image_data.append(cv.resize(img_positive,(IMG_W,IMG_H),interpolation=cv.INTER_CUBIC))
    image_data.append(cv.resize(img_negative, (IMG_W, IMG_H), interpolation=cv.INTER_CUBIC))
    image_data.append(cv.resize(img_negative2, (IMG_W, IMG_H), interpolation=cv.INTER_CUBIC))

    image_data =np.array(image_data,dtype='float32')
    image_data =(image_data)/256.0
    # anchor_score = face_val(image_data[0],False)
    # print(anchor_score)
    pos_d,neg_d =face_val(image_data,True)

    print('相同',pos_d,'不同',neg_d,'距离差',neg_d-pos_d)



    cv.imshow('positive', sh_positive)
    cv.imshow('negative', sh_negative)
    cv.waitKey()
    cv.destroyAllWindows()

# X_data = get_img_data()
# for i in range(20):
#     i = i + 2624
#     pos_d, neg_d = face_val(X_data[i], True)
#
#
#     print('相同', pos_d, '不同', neg_d, '距离差', neg_d - pos_d)
#
#     cv.imshow('anchor', X_data[i][0]*256)
#     cv.waitKey()
#     cv.destroyAllWindows()








