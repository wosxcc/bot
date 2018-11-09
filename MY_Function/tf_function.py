import tensorflow as tf
import numpy as np
from scipy import misc
from tensorflow.python.ops import control_flow_ops

RANDOM_ROTATE = 1           # 随机旋转
RANDOM_CROP = 2             # 随机裁剪
RANDOM_FLIP = 4             # 随机水平翻转
FIXED_STANDARDIZATION = 8   # 归一化运算
FLIP = 16
# def create_input_pipeline(filename, label, control, image_size,nrof_preprocess_threads, batch_size_placeholder):
#     images_and_labels_list = []
#     print('filenames', filename)
#     print('label', label)
#     print('control', control)
#     file_contents = tf.read_file(filename)
#     image = tf.image.decode_image(file_contents, 3)
#
#     image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),            # 随机旋转
#                     lambda:tf.py_func(random_rotate_image, [image], tf.uint8),
#                     lambda:tf.identity(image))
#     image = tf.cond(get_control_flag(control[0], RANDOM_CROP),              # 随机裁剪
#                     lambda:tf.random_crop(image, image_size + (3,)),
#                     lambda:tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
#     image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),              # 随机水平翻转
#                     lambda:tf.image.random_flip_left_right(image),
#                     lambda:tf.identity(image))
#     image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),    # 归一化运算
#                     lambda:(tf.cast(image, tf.float32) - 127.5)/128.0,
#                     lambda:tf.image.per_image_standardization(image))
#     image = tf.cond(get_control_flag(control[0], FLIP),                     # 图像水平翻转
#                     lambda:tf.image.flip_left_right(image),
#                     lambda:tf.identity(image))
#     image.set_shape(image_size + (3,))
#     # images_and_labels_list.append([image, label])
#     image_batch, label_batch = tf.train.batch([image, label],
#                                               batch_size=batch_size_placeholder,
#                                               num_threads=32,
#                                               capacity=4 * 100)
#     return image_batch, label_batch

def create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder):
    # print(image_size)
    # print(nrof_preprocess_threads)
    images_and_labels_list = []
    for _ in range(nrof_preprocess_threads):
        filenames, label, control = input_queue.dequeue()
        images = []
        # print('control',control)
        # print('看看你干的好事filenames', filenames)
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, 3)


            image = tf.cond(get_control_flag(control[0], RANDOM_ROTATE),            # 随机旋转
                            lambda:tf.py_func(random_rotate_image, [image], tf.uint8),
                            lambda:tf.identity(image))

            # print('去去去去', image)
            image = tf.cond(get_control_flag(control[0], RANDOM_CROP),              # 随机裁剪
                            lambda:tf.random_crop(image, image_size + (3,)),
                            lambda:tf.image.resize_image_with_crop_or_pad(image, image_size[0], image_size[1]))
            image = tf.cond(get_control_flag(control[0], RANDOM_FLIP),              # 随机水平翻转
                            lambda:tf.image.random_flip_left_right(image),
                            lambda:tf.identity(image))
            image = tf.cond(get_control_flag(control[0], FIXED_STANDARDIZATION),    # 归一化运算
                            lambda:(tf.cast(image, tf.float32) - 127.5)/128.0,
                            lambda:tf.image.per_image_standardization(image))
            image = tf.cond(get_control_flag(control[0], FLIP),                     # 图像水平翻转
                            lambda:tf.image.flip_left_right(image),
                            lambda:tf.identity(image))
            #pylint: disable=no-member
            image.set_shape(image_size + (3,))
            images.append(image)
        images_and_labels_list.append([images, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels_list, batch_size=batch_size_placeholder,
        shapes=[image_size + (3,), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * 100,
        allow_smaller_final_batch=True)
    # print('回到了原点了啊 哈哈哈哈',image_batch, label_batch)
    return image_batch, label_batch


def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')
def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)






def face_net(image_batch,face_class,is_train=True):
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


    def weight_variable(shape,name='weight'):
        initial = tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32)
        return tf.get_variable(name,shape=shape,dtype=tf.float32,initializer=initial)

    def bias_variable(shape,name='biases'):
        initial =tf.constant_initializer(0.1)
        return tf.get_variable(name,shape=shape,dtype=tf.float32,initializer=initial)


    with tf.variable_scope('conv1') as scope:
        W1 = weight_variable([11, 11, 3, 32])
        b1 = bias_variable([32])
        conv = tf.nn.conv2d(image_batch, W1, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, b1)
        relu1 = batch_norm(tf.nn.relu(pre_activation, name="relu1"),is_train)

    with tf.variable_scope('conv2') as scope:
        W2 = weight_variable([3, 3, 32, 64])
        b2 = bias_variable([64])
        conv2 = tf.nn.conv2d(relu1, W2, strides=[1, 2, 2, 1], padding='SAME')
        relu2 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv2, b2), name='relu2'),is_train)

    with tf.variable_scope('conv3') as scope:
        W3 = weight_variable([9, 9, 64, 16])
        b3 = bias_variable([16])
        conv3 = tf.nn.conv2d(relu2, W3, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv3, b3), name='relu3'),is_train)

    with tf.variable_scope('conv4') as scope:
        W4 = weight_variable([9, 9, 16, 32])
        b4 = bias_variable([32])
        conv4 = tf.nn.conv2d(relu3, W4, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv4, b4), name='relu4'),is_train)

    with tf.variable_scope('conv5') as scope:
        W5 = weight_variable([3, 3, 32, 256])
        b5 = bias_variable([256])
        conv5 = tf.nn.conv2d(relu4, W5, strides=[1, 2, 2, 1], padding='SAME')
        relu5 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv5, b5), name='relu5'),is_train)

    with tf.variable_scope('conv6') as scope:
        W6 = weight_variable([7, 7, 256, 32])
        b6 = bias_variable([32])
        conv6 = tf.nn.conv2d(relu5, W6, strides=[1, 1,1, 1], padding='SAME')
        relu6 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv6, b6), name='relu6'),is_train)

    with tf.variable_scope('conv7') as scope:
        W7 = weight_variable([5, 5, 32, 32])
        b7= bias_variable([32])
        conv7 = tf.nn.conv2d(relu6, W7, strides=[1, 1, 1, 1], padding='SAME')
        relu7 = batch_norm(tf.nn.relu(tf.nn.bias_add(conv7, b7), name='relu7'),is_train)

        # 全连接层
    with tf.variable_scope("fc1") as scope:
        dim = int(np.prod(relu7.get_shape()[1:]))
        reshape = tf.reshape(relu7, [-1, dim])
        weights1 =weight_variable([dim, 256])   ##24*24*256*256
        biases1 = bias_variable([256])
        fc1 = batch_norm(tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, weights1) + biases1, name="fc1"),0.5),is_train)

    with tf.variable_scope("output") as scope:
        weights2 = weight_variable([256, face_class])
        biases2 = bias_variable([face_class])
        y_conv=tf.add(tf.matmul(fc1, weights2),biases2, name="output")
    return y_conv