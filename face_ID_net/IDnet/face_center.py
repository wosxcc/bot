from scipy import misc
import os
import  cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
RANDOM_ROTATE = 1           # 随机旋转
RANDOM_CROP = 2             # 随机裁剪
RANDOM_FLIP = 4             # 随机水平翻转
FIXED_STANDARDIZATION = 8   # 归一化运算
FLIP = 16                   # 图像水平翻转

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


def get_data_img():
    img_path = 'E:/about_Face/facenet-master/data/casia_maxpy_mtcnnpy_182'
    face_img = []
    face_lab = []
    count = 0
    for spath in os.listdir(img_path):
        for file in os.listdir(img_path + '/' + spath):
            face_img.append(img_path + '/' + spath + '/' + file)
            face_lab.append(count)
        count += 1
    # face_img = np.array(face_img, dtype='str')
    # face_lab = np.array(face_lab, dtype='int')
    return face_img,face_lab



def face_id_net(image_batch,face_class,is_train=True):


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
        weights1 =weight_variable([dim, 1024])   ##24*24*256*256
        biases1 = bias_variable([1024])
        fc1 = batch_norm(tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, weights1) + biases1, name="fc1"),0.5),is_train)

    with tf.variable_scope("output") as scope:
        weights2 = weight_variable([1024, face_class])
        biases2 = bias_variable([face_class])
        y_conv=tf.add(tf.matmul(fc1, weights2),biases2, name="output")
    return y_conv

def center_loss(net,label_batch,alfa,nclass):
    norf_net  =net.get_shape()[1]
    centers = tf.get_variable('centers',[nclass,norf_net],dtype=tf.float32,
                              initializer=tf.constant_initializer(0),trainable=False)

    label = tf.reshape(label_batch,[-1])
    centers_batch = tf.gather(centers,label) #从'centers'根据'label'的参数值获取切片。就是在axis维根据indices取某些值。

    diff = (1-alfa)*(centers_batch-net)
    centers = tf.scatter_sub(centers,label,diff) #对centers中的label位置的数据减去diff

    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(net-centers_batch))
    return loss, centers

def trian_optimizer(tloss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(tloss, global_step=global_step)
    return train_op


#                                                       预处理线程
def train_face_net(img_data,lab_data,nclass,image_size,nrof_preprocess_threads,face_class,alfa,max_epoch,batch_size,epoch_num,learning_rate=0.0001):
    read_train_dir = './botid/1010/'
    seave_train_dir = './botid/1010/'
    batch_size_placeholder =tf.placeholder(tf.int32,name='batch_size')

    phase_train_placeholder = tf.placeholder(tf.bool,name='phase_train')

    image_paths_placeholder = tf.placeholder(tf.string,shape=(None,1),name='image_paths')

    labels_placeholder = tf.placeholder(tf.int32,shape=(None,1),name='labels')

    control_placeholder = tf.placeholder(tf.int32,shape=(None,1),name='control')


    input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                    dtypes=[tf.string, tf.int32, tf.int32],
                                    shapes=[(1,), (1,), (1,)],
                                    shared_name=None, name=None)
    enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')

    image_batch, label_batch = create_input_pipeline(input_queue, image_size, nrof_preprocess_threads,batch_size_placeholder)

    image_batch = tf.identity(image_batch, 'image_batch')
    image_batch = tf.identity(image_batch, 'input')
    label_batch = tf.identity(label_batch, 'label_batch')

    labels = ops.convert_to_tensor(lab_data, dtype=tf.int32)
    range_size = array_ops.shape(labels)[0]
    index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                shuffle=True, seed=None, capacity=32)
    index_dequeue_op = index_queue.dequeue_many(batch_size * epoch_num, 'index_dequeue')
    prelogits = face_id_net(image_batch, face_class, phase_train_placeholder)
    prelogits_center_loss,_ = center_loss(prelogits, label_batch, alfa, nclass)

    logits = slim.fully_connected(prelogits, nclass, activation_fn=None,
                                  weights_initializer=slim.initializers.xavier_initializer(),
                                  weights_regularizer=slim.l2_regularizer(0.0005),
                                  scope='Logits', reuse=False)
    eps = 1e-4
    prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=1, axis=1))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * 0.0005)

    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * 0.0)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_batch, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # Calculate the total losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

    train_op = trian_optimizer(total_loss,learning_rate=learning_rate)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)



    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(read_train_dir)
    y_step = 0
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('原有训练次数', global_step)
        y_step = int(float(global_step))

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)
    with sess.as_default():
        for epoch in range(1,max_epoch+1):

            # print('你说这是什么事:img_data',type(img_data),len(img_data),img_data)
            # print('你说这是什么事:lab_data',type(lab_data), len(lab_data), lab_data)
            index_epoch = sess.run(index_dequeue_op)
            # print(index_epoch)
            label_epoch = np.array(lab_data)[index_epoch]
            image_epoch = np.array(img_data)[index_epoch]

            # Enqueue one epoch of image paths and labels
            labels_array = np.expand_dims(np.array(label_epoch), 1)
            # print('这里的图片是什么', image_epoch.shape, image_epoch)
            # print('到这里标签变成什么了', labels_array.shape, labels_array)
            image_paths_array = np.expand_dims(np.array(image_epoch), 1)
            control_value =14
            control_array = np.ones_like(labels_array) * control_value
            sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                                  control_placeholder: control_array})

            batch_number =0
            while batch_number < epoch_num:
                tensor_list = [total_loss, train_op,  regularization_losses, prelogits, cross_entropy_mean,prelogits_norm, accuracy, prelogits_center_loss]
                feed_dict={phase_train_placeholder:True, batch_size_placeholder:batch_size}
                loss_, _,  reg_losses_, prelogits_, cross_entropy_mean_, prelogits_norm_, accuracy_, center_loss_=sess.run(tensor_list , feed_dict=feed_dict)

                print('Epoch: [%d][%d/%d]\t总loss %2.3f\t交叉熵平均 %2.3f\t正则化loss %2.3f\t准确率 %2.3f\t中心loss %2.3f' %
                    (epoch, batch_number + 1, epoch_num, loss_, cross_entropy_mean_, np.sum(reg_losses_),accuracy_, center_loss_))
                batch_number+=1

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
                                                                       ['output/output'])
            with tf.gfile.FastGFile(seave_train_dir + 'face72.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            checkpoint_path = os.path.join(seave_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch + y_step)
    return True

img_data,lab_data = get_data_img()
epoch_size =1000
# img_data,lab_data
nclass =int(lab_data[-1]+1)
batch_size =30
image_size = (160, 160)
nrof_preprocess_threads = 4
epoch_num = 1000
face_class = 512
alfa = 0.95
max_epoch = 200
learning_rate=0.0001
train_face_net(img_data,lab_data,nclass,image_size,nrof_preprocess_threads,face_class,alfa,max_epoch,batch_size,epoch_num,learning_rate)
