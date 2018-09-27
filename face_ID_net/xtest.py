import os
import cv2 as cv
import numpy as np
import  random
import tensorflow as tf

from face_ID_net.IDnet import  face_net
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
IMG_H=64
IMG_W =64
N_CLASSES =128
learning_rate =0.001

def face_val(image_arr,modelss):
    print('搞毛线啊')
    log_dir = './face72/faceIDcard/'
    # image_arr=test_file
    with tf.Graph().as_default():
        # image = tf.cast(image_arr, tf.float32)
        # image = tf.image.per_image_standardization(image)  ###归一化操作
        # image = tf.reshape(image_arr, [1,3, 64, 64, 3])
        graph = face_net(1, IMG_H,IMG_W, N_CLASSES,learning_rate,15,modelss)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('没有保存的模型')
            if modelss ==3:
                pos_d,neg_d = sess.run([graph['d_pos'],graph['d_neg']], feed_dict={graph['x']: np.reshape(image_arr, (1,3, 64, 64, 3))})
                return pos_d, neg_d
            elif modelss ==1:
                print('下面出错了',len(image_arr),image_arr[0].shape)

                anchor_data = sess.run(graph['anchor_out'],feed_dict={graph['x']: np.reshape(image_arr, (1, 1, 64, 64, 3))})
                print('上面出错了')
                return anchor_data




pacth = 'E:/faceID'
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

img_anchor=cv.imread(pacth+'/'+file+'/'+anchor_img)
img_positive=cv.imread(pacth+'/'+file+'/'+positive_img)
img_negative=cv.imread(pacth+'/'+negative_file+'/'+negative_img)


sh_anchor=cv.resize(img_anchor,(240,240),interpolation=cv.INTER_CUBIC)
sh_positive=cv.resize(img_positive,(240,240),interpolation=cv.INTER_CUBIC)
sh_negative=cv.resize(img_negative,(240,240),interpolation=cv.INTER_CUBIC)

image_data=[]

image_data.append(cv.resize(img_anchor,(64,64),interpolation=cv.INTER_CUBIC))
image_data.append(cv.resize(img_positive,(64,64),interpolation=cv.INTER_CUBIC))
image_data.append(cv.resize(img_negative,(64,64),interpolation=cv.INTER_CUBIC))

image_data =np.array(image_data,dtype='float32')
image_data =(image_data-128.0)/256.0
anchor_score = face_val(image_data[0],1)
print(anchor_score)
pos_d,neg_d =face_val(image_data,3)

print(pos_d,neg_d)




cv.imshow('anchor', sh_anchor)
cv.imshow('positive', sh_positive)
cv.imshow('negative', sh_negative)
cv.waitKey()
cv.destroyAllWindows()





# def face_net(batch_size,height, width, n_classes,learning_rate,margin,image_count=3):
#     print(batch_size,height, width, n_classes,learning_rate)
#     x = tf.placeholder(tf.float32, shape=[batch_size,3, height, width, 3], name='input')
#     # y = tf.placeholder(tf.float32, shape=[None, n_classes], name='labels')
#
#     def weight_variable(shape, name="weights"):
#         initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
#         return tf.Variable(initial, name=name)
#
#     def bias_variable(shape, name="biases"):
#         initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
#         return tf.Variable(initial, name=name)
#
#     for xx in range(image_count):
#         xnow_x =tf.slice(x, [0, xx, 0, 0, 0], [batch_size, 1,height, width, 3])
#         now_x = tf.reshape(xnow_x, shape=[batch_size, height, width, 3], name=None)
#         with tf.variable_scope('conv1') as scope:
#             W1 = weight_variable([3, 3, 3, 32])
#             b1 = bias_variable([32])
#             conv = tf.nn.conv2d(now_x, W1, strides=[1, 1, 1, 1], padding="SAME")
#             pre_activation = tf.nn.bias_add(conv, b1)
#             relu1 = tf.nn.relu(pre_activation, name="relu1")
#
#         with tf.variable_scope('conv2') as scope:
#             W2 = weight_variable([3, 3, 32, 64])
#             b2 = bias_variable([64])
#             conv2 = tf.nn.conv2d(relu1, W2, strides=[1, 2, 2, 1], padding='SAME')
#             relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name='relu2')
#
#         with tf.variable_scope('conv3') as scope:
#             W3 = weight_variable([3, 3, 64, 128])
#             b3 = bias_variable([128])
#             conv3 = tf.nn.conv2d(relu2, W3, strides=[1, 1, 1, 1], padding='SAME')
#             relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name='relu3')
#
#         with tf.variable_scope('conv4') as scope:
#             W4 = weight_variable([3, 3, 128, 256])
#             b4 = bias_variable([256])
#             conv4 = tf.nn.conv2d(relu3, W4, strides=[1, 2, 2, 1], padding='SAME')
#             relu4 = tf.nn.relu(tf.nn.bias_add(conv4, b4), name='relu4')
#
#
#         with tf.variable_scope('conv5') as scope:
#             W5 = weight_variable([3, 3, 256, 128])
#             b5 = bias_variable([128])
#             conv5 = tf.nn.conv2d(relu4, W5, strides=[1, 1, 1, 1], padding='SAME')
#             relu5 = tf.nn.relu(tf.nn.bias_add(conv5, b5), name='relu5')
#
#         relu66 = concatenate([ relu3, UpSampling2D(2)(relu5)])
#         # print('看看是什么',relu66)
#         with tf.variable_scope('conv7') as scope:
#             W7 = weight_variable([3, 3, 256, 128])
#             b7= bias_variable([128])
#             conv7 = tf.nn.conv2d(relu66, W7, strides=[1, 2, 2, 1], padding='SAME')
#             relu7 = tf.nn.relu(tf.nn.bias_add(conv7, b7), name='relu7')
#
#             # 全连接层
#         with tf.variable_scope("fc1") as scope:
#
#             dim = int(np.prod(relu7.get_shape()[1:]))
#             reshape = tf.reshape(relu7, [-1, dim])
#             weights1 =weight_variable([dim, 300])
#             biases1 = bias_variable([300])
#             fc1 = tf.nn.relu(tf.matmul(reshape, weights1) + biases1, name="fc1")
#
#         with tf.variable_scope("output") as scope:
#             weights2 = weight_variable([300, n_classes])
#             biases2 = bias_variable([n_classes])
#             y_conv = tf.add(tf.matmul(fc1, weights2), biases2, name="output")
#             y_conv =tf.tanh(y_conv,name="output")
#         if xx == 0:
#             anchor_out= y_conv
#         elif xx == 1:
#             positive_out= y_conv
#         elif xx == 2:
#             negative_out= y_conv
#
#     if image_count==3:
#         # d_pos = tf.reduce_sum(tf.square(anchor_out - positive_out), 1)
#         d_pos = tf.norm(anchor_out - positive_out, axis=1)
#         print('搞什么毛线d_pos',d_pos)
#         # d_neg = tf.reduce_sum(tf.square(anchor_out - negative_out), 1)
#         d_neg = tf.norm(anchor_out - negative_out, axis=1)
#         loss = tf.maximum(0.0, margin + d_pos - d_neg)
#         print('你这是干什么',loss)
#         loss = tf.reduce_mean(loss) + tf.reduce_mean(d_pos)
#         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#         global_step = tf.Variable(0, name="global_step", trainable=False)
#         train_op = optimizer.minimize(loss, global_step=global_step)
#
#         return dict(
#             x=x,
#             loss=loss,
#             optimize=train_op,
#             d_pos=d_pos,
#             d_neg=d_neg,
#         )
#     if image_count==1:
#         return dict(
#             x=x,
#             anchor_out=anchor_out,
#         )
#





