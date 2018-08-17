import tensorflow as  tf
import  numpy as np

batch_size = 64
img_size=96
valid_data=[]
test_data=[]
class_num =1

learning_rate =0.001
momentum_param=0.8

tf_train_data=tf.placeholder(tf.float32,shape=(batch_size,img_size,img_size,3))
tf_train_label=tf.placeholder(tf.float32,shape=(batch_size,class_num))
tf_valid_data=tf.constant(valid_data)
tf_test_data=tf.constant(test_data)

w = {
    'w1': tf.get_variable('w1', [3, 3, 3, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w2': tf.get_variable('w2', [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w3': tf.get_variable('w3', [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w4': tf.get_variable('w4', [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w5': tf.get_variable('w5', [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w6': tf.get_variable('w6', [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w7': tf.get_variable('w7', [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w8': tf.get_variable('w8', [3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w9': tf.get_variable('w9', [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w10': tf.get_variable('w10', [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w11': tf.get_variable('w11', [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w12': tf.get_variable('w12', [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w13': tf.get_variable('w13', [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer_conv2d()),
    'w14': tf.Variable(tf.random_normal([img_size / 32 * img_size / 32 * 512, 4096])),
    'w15': tf.Variable(tf.random_normal([4096, 4096])),
    'w16': tf.Variable(tf.random_normal([4096, 1000])),
}

b = {
    'b1': tf.Variable(tf.zeros([64])),
    'b2': tf.Variable(tf.zeros([64])),
    'b3': tf.Variable(tf.zeros([128])),
    'b4': tf.Variable(tf.zeros([128])),
    'b5': tf.Variable(tf.zeros([256])),
    'b6': tf.Variable(tf.zeros([256])),
    'b7': tf.Variable(tf.zeros([256])),
    'b8': tf.Variable(tf.zeros([512])),
    'b9': tf.Variable(tf.zeros([512])),
    'b10': tf.Variable(tf.zeros([512])),
    'b11': tf.Variable(tf.zeros([512])),
    'b12': tf.Variable(tf.zeros([512])),
    'b13': tf.Variable(tf.zeros([512])),
    'b14': tf.Variable(tf.zeros([4096])),
    'b15': tf.Variable(tf.zeros([4096])),
    'b16': tf.Variable(tf.zeros([1000])),
}

drop_param =0.5

def model(input_data):
    conv1 = tf.nn.conv2d(input_data, w['w1'], [1, 1, 1, 3], padding="SAME")  # img:224*224*64
    h1 = tf.nn.relu(conv1 + b['b1'])
    conv2 = tf.nn.conv2d(h1, w['w2'], [1, 1, 1, 64], padding="SAME")
    h2 = tf.nn.relu(conv2 + b['b2'])

    max1 = tf.nn.max_pool(h2, [2, 2, 64, 64], [1, 2, 2, 64], padding="VALID")

    conv3 = tf.nn.conv2d(max1, w['w3'], [1, 1, 1, 64], padding="SAME")  # img:112*112*128
    h3 = tf.nn.relu(conv3 + b['b3'])
    conv4 = tf.nn.conv2d(h3, w['w4'], [1, 1, 1, 128], padding="SAME")
    h4 = tf.nn.relu(conv4 + b['b4'])

    max2 = tf.nn.max_pool(h4, [2, 2, 128, 128], [1, 2, 2, 128], padding="VALID")

    conv5 = tf.nn.conv2d(max2, w['w5'], [1, 1, 1, 128], padding="SAME")  # img:56*56*256
    h5 = tf.nn.relu(conv5 + b['b5'])
    conv6 = tf.nn.conv2d(h5, w['w6'], [1, 1, 1, 256], padding="SAME")
    h6 = tf.nn.relu(conv6 + b['b6'])
    conv7 = tf.nn.conv2d(h6, w['w7'], [1, 1, 1, 256], padding="SAME")
    h7 = tf.nn.relu(conv7) + b['b7']

    max3 = tf.nn.max_pool(h7, [2, 2, 256, 256], [1, 2, 2, 256], padding="VALID")

    conv8 = tf.nn.conv2d(max3, w['w8'], [1, 1, 1, 256], padding="SAME")  # img:28*28*512
    h8 = tf.nn.relu(conv8 + b['b8'])
    conv9 = tf.nn.conv2d(h8, w['w9'], [1, 1, 1, 512], padding="SAME")
    h9 = tf.nn.relu(conv9 + b['b9'])
    conv10 = tf.nn.conv2d(h9, w['w10'], [1, 1, 1, 512], padding="SAME")
    h10 = tf.nn.relu(conv10 + b['b10'])

    max4 = tf.nn.max_pool(h10, [2, 2, 512, 512], [1, 2, 2, 512], padding="VALID")

    conv11 = tf.nn.conv2d(max4, w['w11'], [1, 1, 1, 512], padding="SAME")  # img:14*14*512
    h11 = tf.nn.relu(conv11 + b['b11'])
    conv12 = tf.nn.conv2d(h11, w['w12'], [1, 1, 1, 512], padding="SAME")
    h12 = tf.nn.relu(conv12 + b['b12'])
    conv13 = tf.nn.conv2d(h12, w['w13'], [1, 1, 1, 512], padding="SAME")
    h13 = tf.nn.relu(conv13 + b['b13'])

    max5 = tf.nn.max_pool(h13, [2, 2, 512, 512], [1, 2, 2, 512], padding="VALID")

    shapes = max5.get_shape().as_list()  # img:7*7*512
    reshape = tf.reshape(max5, [shapes[0], shapes[1] * shapes[2] * shapes[3]])
    fc1 = tf.matmul(reshape, w['w14']) + b['b14']
    h14 = tf.nn.dropout(tf.nn.relu(fc1), drop_param)
    fc2 = tf.matmul(h14, w['w15']) + b['b15']
    h15 = tf.nn.dropout(tf.nn.relu(fc2), drop_param)
    fc3 = tf.matmul(h15, w['w16']) + b['b16']
    return fc3

y_yuce =model(tf_train_data)
logits = tf.sqrt(tf.reduce_mean(tf.square(y_yuce - tf_train_label)))


l2_loss=None
l2_param =0.001

train_predictions = tf.nn.softmax(logits)
valid_predictions = tf.nn.softmax(model(tf_valid_data))
test_predictions = tf.nn.softmax(model(tf_test_data))
for i in np.arange(1,17):
    k="w"+str(i)
    l2_loss+=l2_param*tf.nn.l2_loss(w[k])
    loss=tf.reduce_mean(tf.softmax_cross_entropy_with_logits(logits,tf_train_label))+l2_loss

optimizer=tf.train.MomentumOptimizer(learning_rate,momentum_param)




# from  datetime import datetime
# import tensorflow as tf
# import math
# import time
#
# batch_size = 32
# num_batches = 100
#
# # 用来创建卷积层并把本层的参数存入参数列表
# # input_op:输入的tensor name:该层的名称 kh:卷积层的高 kw:卷积层的宽 n_out:输出通道数，dh:步长的高 dw:步长的宽，p是参数列表
# def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
#     # 输入的通道数
#     n_in = input_op.get_shape()[-1].value
#     with tf.name_scope(name) as scope:
#         kernel = tf.get_variable(scope + "w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
#         conv = tf.nn.conv2d(input_op, kernel, (1,dh,dw,1),padding='SAME')
#         bias_init_val = tf.constant(0.0, shape=[n_out],dtype=tf.float32)
#         biases = tf.Variable(bias_init_val , trainable=True , name='b')
#         z = tf.nn.bias_add(conv,biases)
#         activation = tf.nn.relu(z,name=scope)
#         p += [kernel,biases]
#         return activation
#
# # 定义全连接层
# def fc_op(input_op,name,n_out,p):
#     n_in = input_op.get_shape()[-1].value
#     with tf.name_scope(name) as scope:
#         kernel = tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
#         biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
#         # tf.nn.relu_layer()用来对输入变量input_op与kernel做乘法并且加上偏置b
#         activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
#         p += [kernel,biases]
#         return activation
#
# # 定义最大池化层
# def mpool_op(input_op,name,kh,kw,dh,dw):
#     return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)
#
# #定义网络结构
# def inference_op(input_op,keep_prob):
#     p = []
#     conv1_1 = conv_op(input_op,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
#     conv1_2 = conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
#     pool1 = mpool_op(conv1_2,name='pool1',kh=2,kw=2,dw=2,dh=2)
#
#     conv2_1 = conv_op(pool1,name='conv2_1',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
#     conv2_2 = conv_op(conv2_1,name='conv2_2',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
#     pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)
#
#     conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
#     conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
#     conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
#     pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dw=2, dh=2)
#
#     conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
#     conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
#     conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
#     pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dw=2, dh=2)
#
#     conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
#     conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
#     conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
#     pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dw=2, dh=2)
#
#     shp = pool5.get_shape()
#     flattened_shape = shp[1].value * shp[2].value * shp[3].value
#     resh1 = tf.reshape(pool5,[-1,flattened_shape],name="resh1")
#
#     fc6 = fc_op(resh1,name="fc6",n_out=4096,p=p)
#     fc6_drop = tf.nn.dropout(fc6,keep_prob,name='fc6_drop')
#     fc7 = fc_op(fc6_drop,name="fc7",n_out=4096,p=p)
#     fc7_drop = tf.nn.dropout(fc7,keep_prob,name="fc7_drop")
#     fc8 = fc_op(fc7_drop,name="fc8",n_out=1000,p=p)
#     softmax = tf.nn.softmax(fc8)
#     predictions = tf.argmax(softmax,1)
#     return predictions,softmax,fc8,p
#
# def time_tensorflow_run(session,target,feed,info_string):
#     num_steps_burn_in = 10  # 预热轮数
#     total_duration = 0.0  # 总时间
#     total_duration_squared = 0.0  # 总时间的平方和用以计算方差
#     for i in range(num_batches + num_steps_burn_in):
#         start_time = time.time()
#         _ = session.run(target,feed_dict=feed)
#         duration = time.time() - start_time
#         if i >= num_steps_burn_in:  # 只考虑预热轮数之后的时间
#             if not i % 10:
#                 print('%s:step %d,duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
#                 total_duration += duration
#                 total_duration_squared += duration * duration
#     mn = total_duration / num_batches  # 平均每个batch的时间
#     vr = total_duration_squared / num_batches - mn * mn  # 方差
#     sd = math.sqrt(vr)  # 标准差
#     print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' % (datetime.now(), info_string, num_batches, mn, sd))
#
# def run_benchmark():
#     with tf.Graph().as_default():
#         image_size = 224  # 输入图像尺寸
#         images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
#         keep_prob = tf.placeholder(tf.float32)
#         prediction,softmax,fc8,p = inference_op(images,keep_prob)
#         init = tf.global_variables_initializer()
#         sess = tf.Session()
#         sess.run(init)
#         time_tensorflow_run(sess, prediction,{keep_prob:1.0}, "Forward")
#         # 用以模拟训练的过程
#         objective = tf.nn.l2_loss(fc8)  # 给一个loss
#         grad = tf.gradients(objective, p)  # 相对于loss的 所有模型参数的梯度
#         time_tensorflow_run(sess, grad, {keep_prob:0.5},"Forward-backward")
#
#
# run_benchmark()