import tensorflow as tf
import numpy as np
import os
import math
import cv2 as cv

train_dir = 'E:/BaiduNetdiskDownload/Dogs vs Cats Redux Kernels Edition'
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


def tf_layers(images, batch_size, n_classes):
    W1 = tf.get_variable('weights1', shape=(3, 3, 3, 32), dtype=tf.float32
                         , initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b1= tf.get_variable('biases1',shape=[32],dtype=tf.float32
                        ,initializer=tf.constant_initializer(0.1))
    conv1 = tf.nn.conv2d(images, W1, strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu( tf.nn.bias_add(conv1, b1), name='relu1')

    W2 = tf.get_variable('weights2', shape=(3, 3, 32, 64), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b2 = tf.get_variable('biases2', shape=[64], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv2 = tf.nn.conv2d(relu1, W2, strides=[1, 2, 2, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name='relu2')

    W3 = tf.get_variable('weights3', shape=(3, 3, 64, 128), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b3 = tf.get_variable('biases3', shape=[128], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv3 = tf.nn.conv2d(relu2, W3, strides=[1, 1, 1, 1], padding='SAME')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name='relu3')

    W4 = tf.get_variable('weights4', shape=(3, 3, 128, 256), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b4 = tf.get_variable('biases4', shape=[256], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv4 = tf.nn.conv2d(relu3, W4, strides=[1, 2, 2, 1], padding='SAME')
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, b4), name='relu4')

    W5 = tf.get_variable('weights5', shape=(3, 3, 256, 256), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b5 = tf.get_variable('biases5', shape=[256], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv5 = tf.nn.conv2d(relu4, W5, strides=[1, 2, 2, 1], padding='SAME')
    relu5 = tf.nn.relu(tf.nn.bias_add(conv5, b5), name='relu5')

    W6 = tf.get_variable('weights6', shape=(3, 3, 256, 256), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b6 = tf.get_variable('biases6', shape=[256], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv6 = tf.nn.conv2d(relu5, W6, strides=[1, 2, 2, 1], padding='SAME')
    relu6 = tf.nn.relu(tf.nn.bias_add(conv6, b6), name='relu6')

    W7 = tf.get_variable('weights7', shape=(1, 1, 256, 128), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b7 = tf.get_variable('biases7', shape=[128], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv7 = tf.nn.conv2d(relu6, W7, strides=[1, 1, 1, 1], padding='SAME')
    relu7 = tf.nn.relu(tf.nn.bias_add(conv7, b7), name='relu7')

    W8 = tf.get_variable('weights8', shape=(1, 1, 128, 128), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b8 = tf.get_variable('biases8', shape=[128], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv8 = tf.nn.conv2d(relu7, W8, strides=[1, 1, 1, 1], padding='SAME')
    relu8 = tf.nn.relu(tf.nn.bias_add(conv8, b8), name='relu8')

    W9 = tf.get_variable('weights9', shape=(1, 1, 128, 64), dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b9 = tf.get_variable('biases9', shape=[64], dtype=tf.float32
                         , initializer=tf.constant_initializer(0.1))
    conv9 = tf.nn.conv2d(relu8, W9, strides=[1, 1, 1, 1], padding='SAME')
    relu9 = tf.nn.relu(tf.nn.bias_add(conv9, b9), name='relu9')

    reshape = tf.reshape(relu9, shape=[batch_size, -1])
    dim = reshape.get_shape()[1].value
    W10 = tf.get_variable('weights10', shape=[dim, n_classes], dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    biases = tf.get_variable("biases",
                             shape=[n_classes],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    softmax_linear = tf.add(tf.matmul(reshape, W10), biases, name="softmax_linear")
    return softmax_linear


###定义损失函数，返回损失值
def losses(logits, lables):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=lables, name="xentropy_per_example")
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
    correct = tf.nn.in_top_k(logits, labels, k=1)
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    return accuracy


def run_training():
    logs_train_dir = './mycnn/log/'
    train,train_label=get_files('E:/BaiduNetdiskDownload/Dogs vs Cats Redux Kernels Edition',0.01)

    train_batch,train_label_batch=get_batch(train,train_label,IMG_W,IMG_H,BATCH_SIZE,
                                      CAPACITY)
    train_logits = tf_layers(train_batch,BATCH_SIZE,N_CLASSES)
    train_loss= losses(train_logits,train_label_batch)
    train_op =trainning(train_loss,learning_rate=learning_rete)
    train_acc =evaluation(train_logits,train_label_batch)

    summary_op =tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess =tf.Session(config=config)
    train_writer=tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver= tf.train.Saver()

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



N_CLASSES = 2
IMG_W = 48
IMG_H = 48
BATCH_SIZE = 100
CAPACITY = 64
MAX_STEP = 10000
learning_rete =0.0001
run_training()





