# coding:utf-8
import collections
import tensorflow as tf
import time
import math
import os
import numpy as np
from datetime import datetime
slim = tf.contrib.slim






class Block(collections.namedtuple('Bolck', ['scope', 'unit_fn', 'args'])):
    'A named tuple describing a ResNet block.'



train_dir = 'D:/pproject/ppop/my_tf/cat_dog'
husky = []
label_husky = []
jiwawa = []
label_jiwawa = []


# label_qiutian = []


# step1：获取'E:/Re_train/image_data/training_image'下所有的图片路径名，存放到
# 对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir, ratio):
    for file in os.listdir(file_dir + '/dog'):
        husky.append(file_dir + '/dog' + '/' + file)
        label_husky.append(0)
    for file in os.listdir(file_dir + '/cat'):
        jiwawa.append(file_dir + '/cat' + '/' + file)
        label_jiwawa.append(1)
    # step2：对生成的图片路径和标签List做打乱处理把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((husky, jiwawa))
    label_list = np.hstack((label_husky, label_jiwawa))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    # ratio是测试集的比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 测试样本数
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels  # , val_images, val_labels


# ---------------------------------------------------------------------------
# --------------------生成Batch----------------------------------------------

# step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
# 是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
#   image_W, image_H, ：设置好固定的图像高度和宽度
#   设置batch_size：每个batch要放多少张图片
#   capacity：一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)  ##裁剪填补工作
    image = tf.image.random_brightness(image, max_delta=0.5)  ##在-0.5到0.5之间随机调整亮度
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  ###在-0.5到0.5之间随机调整亮度
    image = tf.image.random_hue(image, 0.5)  ##在0-0.5之间随机调整图像饱和度
    # image = tf.image.resize_images(image, [image_W, image_H], method=0)
    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch




def subsample(inputs, factor, scope = None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride = factor, scope = scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope = None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride = 1, padding = 'SAME', scope = scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride = stride, padding = 'VALID', scope = scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections = None):

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' %(i + 1), values = [net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net, depth = unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                    stride = unit_stride)
                    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


def resnet_arg_scope(is_training = True,
                    weight_decay = 0.0001,
                    batch_norm_decay = 0.997,
                    batch_norm_epsilon = 1e-5,
                    batch_norm_scale = True):

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer = slim.l2_regularizer(weight_decay),
            weights_initializer = slim.variance_scaling_initializer(),
            activation_fn = tf.nn.relu,
            normalizer_fn = slim.batch_norm,
            normalizer_params = batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding = 'SAME') as arg_sc:
                return arg_sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections = None, scope = None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank = 4)
        preact = slim.batch_norm(inputs, activation_fn = tf.nn.relu, scope = 'preact')

        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride = stride, normalizer_fn = None, activation_fn = None, scope = 'shortcut')
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride = 1, scope = 'conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope = 'conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride = 1, normalizer_fn = None, activation_fn = None, scope = 'conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)



def resnet_v2(inputs, blocks, num_classes = None, global_pool = True, include_root_block = True, reuse = None, scope = None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse = reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections = end_points_collection):
            net = inputs

        if include_root_block:
            with slim.arg_scope([slim.conv2d], activation_fn = None, normalizer_fn = None):
                net = conv2d_same(net, 64, 7, stride = 2, scope = 'conv1')
            net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'pool1')
        net = stack_blocks_dense(net, blocks)
        net = slim.batch_norm(net, activation_fn = tf.nn.relu, scope = 'postnorm')

        if global_pool:
            net = tf.reduce_mean(net, [1, 2], name = 'pool5', keep_dims = True)

        if num_classes is not None:
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'logits')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope = 'predictions')
            print('net',net)
            net = tf.reshape(net, shape=[BATCH_SIZE, -1])
            return net, end_points



def resnet_v2_50(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_50'):

    print(num_classes, global_pool, reuse)
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 1024, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)


def resnet_v2_101(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_101'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)

def resnet_v2_152(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_152'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)

def resnet_v2_200(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_200'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)


def losses(logits, labels):

    print('logits',logits)
    print('labels',labels)
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name="xe")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
    return loss

def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy

def train_resnet():
    logs_train_dir = './resnet/log/'
    train, train_label = get_files('D:/pproject/ppop/my_tf/cat_dog', 0.2)

    train_batch, train_label_batch = get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE,
                                               CAPACITY)
    print(train_batch, train_label_batch)
    net, end_points = resnet_v2_50(train_batch, N_CLASSES)
    train_loss=losses(net, train_label_batch)
    train_op=trainning(train_loss, learning_rate)
    train_acc = evaluation(net, train_label_batch)
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
            if step % 50 == 0:
                # 每迭代50次，打印出一次结果
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
                print('第{2}次训练loss值是：{0}，准确率为：{1}'.format(sess.run(train_loss),sess.run(train_acc),step))
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



# batch_size = 32
# height, width = 224, 224
# inputs = tf.random_uniform((batch_size, height, width, 3))
# with slim.arg_scope(resnet_arg_scope(is_training = False)):



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
learning_rate=0.01
num_batches = 100

N_CLASSES = 2
IMG_W = 48
IMG_H = 48
BATCH_SIZE = 100
CAPACITY = 64
MAX_STEP = 10000
train_resnet()
