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
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
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
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
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
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
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
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v2(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)

def inception_resnet_v2(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV2'):
    """Creates the Inception Resnet V2 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
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

    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
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
                # 35 x 35 x 192
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_5a_3x3')
                end_points['MaxPool_5a_3x3'] = net

                # 35 x 35 x 320
                with tf.variable_scope('Mixed_5b'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                                    scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                        tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                                    scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                                     scope='AvgPool_0a_3x3')
                        tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                                   scope='Conv2d_0b_1x1')
                    net = tf.concat([tower_conv, tower_conv1_1,
                                     tower_conv2_2, tower_pool_1], 3)

                end_points['Mixed_5b'] = net
                net = slim.repeat(net, 10, block35, scale=0.17)

                # 17 x 17 x 1024
                with tf.variable_scope('Mixed_6a'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',
                                                 scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                                    stride=2, padding='VALID',
                                                    scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                     scope='MaxPool_1a_3x3')
                    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)

                end_points['Mixed_6a'] = net
                net = slim.repeat(net, 20, block17, scale=0.10)

                with tf.variable_scope('Mixed_7a'):
                    with tf.variable_scope('Branch_0'):
                        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                                   padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                                    padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                        tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                                    scope='Conv2d_0b_3x3')
                        tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                                    padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_3'):
                        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                     scope='MaxPool_1a_3x3')
                    net = tf.concat([tower_conv_1, tower_conv1_1,
                                     tower_conv2_2, tower_pool], 3)

                end_points['Mixed_7a'] = net

                net = slim.repeat(net, 9, block8, scale=0.20)
                net = block8(net, activation_fn=None)

                net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
                end_points['Conv2d_7b_1x1'] = net

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
    read_train_dir = './botid/1120/'
    seave_train_dir = './botid/1120/'
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

    prelogits, __ = inference(image_batch, 0.8, phase_train_placeholder,
                            bottleneck_layer_size=face_class, weight_decay=0.001, reuse=None)

    # prelogits = face_id_net(image_batch, face_class, phase_train_placeholder)





    prelogits_center_loss, _ = center_loss(prelogits, label_batch, alfa, nclass)

    logits = slim.fully_connected(prelogits, nclass, activation_fn=None,
                                  weights_initializer=slim.initializers.xavier_initializer(),
                                  weights_regularizer=slim.l2_regularizer(0.0005),
                                  scope='Logits', reuse=False)
    eps = 1e-4
    prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=1, axis=1))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * 0.0005)

    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * 0.00)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_batch, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # Calculate the total losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    print('看看你的真面目',[cross_entropy_mean] ,regularization_losses,prelogits_center_loss)
    total_loss = tf.add_n([cross_entropy_mean] + regularization_losses+[prelogits_center_loss], name='total_loss')

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
                                                                       ['output'])
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
face_class = 256
alfa = 0.95
max_epoch = 200
learning_rate=0.000001
train_face_net(img_data,lab_data,nclass,image_size,nrof_preprocess_threads,face_class,alfa,max_epoch,batch_size,epoch_num,learning_rate)
