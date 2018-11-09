from scipy import misc,stats
import os
import  cv2 as cv
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import random
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from MY_Function.tf_function import create_input_pipeline,face_net
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    return face_img,face_lab


def get_apn_image(embeddings,image_paths,image_label,batch_size,epoch_num):
    apn_img = []
    apn_lab = []
    for i in  range(epoch_num):
        aemb =embeddings[i*batch_size:i*batch_size+batch_size]
        simg_paths = image_paths[i*batch_size:i*batch_size+batch_size]
        slab_paths = image_label[i * batch_size:i * batch_size + batch_size]
        # print('是时候来一波输出')
        # print('aemb',aemb.shape)
        # print('simg_paths',simg_paths)
        # print('slab_paths',slab_paths)

        max_d = 0
        min_d =999999999
        x_a =0
        x_p =0
        x_n =0
        for x in range(batch_size):
            for y in range(batch_size):
                if slab_paths[x] ==slab_paths[y] and x!=y:
                    dist = np.sqrt(np.sum(np.square(aemb[x] - aemb[y])))
                    if max_d<dist:
                        x_a = x
                        x_p = y
                        max_d = dist
        for x in  range(batch_size):
            if slab_paths[x] != slab_paths[x_a]:
                dist = np.sqrt(np.sum(np.square(aemb[x] - aemb[x_a])))
                if min_d>dist:
                    min_d=dist
                    x_n = x
                dist = np.sqrt(np.sum(np.square(aemb[x] - aemb[x_p])))
                if min_d > dist:
                    min_d = dist
                    x_n = x
        # print('你输了么',x_a,x_p,x_n)
        # print(simg_paths[x_a],simg_paths[x_p],simg_paths[x_n])
        apn_img.append([simg_paths[x_a],simg_paths[x_p],simg_paths[x_n]])
        apn_lab.append([slab_paths[x_a],slab_paths[x_p],slab_paths[x_n]])
    return apn_img,apn_lab

def triplet_loss(anchor, positive, negative, alpha):
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        neg_dist2 = tf.reduce_sum(tf.square(tf.subtract(positive, negative)), 1)
        basic_loss1 = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss1 = tf.reduce_sum(tf.maximum(basic_loss1, 0.0), 0)
        basic_loss2= tf.add(tf.subtract(pos_dist, neg_dist2), alpha)
        loss2 = tf.reduce_sum(tf.maximum(basic_loss2, 0.0), 0)
        loss = tf.maximum(loss1,loss2)
    return loss


def _add_loss_summaries(total_loss):
    # 为损失添加摘要。 为损失添加摘要。
    # 为所有损失和相关摘要生成移动平均值
    # 可视化网络的性能。
    # Args:
    # 完全损失：来自损失的全部损失（）。
    # 返回：
    # RuffsIVaReaSeSoop: 用于产生移动平均损失的OP。

    """Add summaries for losses.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.           计算所有个体损失和总损失的移动平均值。
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the      将标量总和附加在所有的个别损失和总损失上；
    # same for the averaged version of the losses.   相同的平均版本的损失。
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss     将每一损失记为“原始”，并列出损失的移动平均版本。
        # as the original loss name.                                                    作为原始损失名称。
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op


def trian_optimizer(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries. 生成所有损失和相关摘要的移动平均值。
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients. 计算梯度
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.                          应用渐变。
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.   为可训练变量添加直方图。
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.             为渐变添加直方图。
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables. 跟踪所有可训练变量的移动平均值。
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

# def trian_optimizer(tloss, learning_rate):
#     with tf.name_scope("optimizer"):
#         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#         global_step = tf.Variable(0, name="global_step", trainable=False)
#         train_op = optimizer.minimize(tloss, global_step=global_step)
#     return train_op
#                                                       预处理线程
def train_face_net(img_data,lab_data,nclass,image_size,nrof_preprocess_threads,face_class,alfa,max_epoch,batch_size,epoch_num,learning_rate=0.0001):
    global_step = tf.Variable(0, trainable=False)
    read_train_dir = './tirplet/1015/'
    seave_train_dir = './tirplet/1015/'
    batch_size_placeholder =tf.placeholder(tf.int32,name='batch_size')
    phase_train_placeholder = tf.placeholder(tf.bool,name='phase_train')
    image_paths_placeholder = tf.placeholder(tf.string,shape=(None,3),name='image_paths')
    labels_placeholder = tf.placeholder(tf.int32,shape=(None,3),name='labels')     #
    control_placeholder = tf.placeholder(tf.int32,shape=(None,3),name='control')   #

    input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                    dtypes=[tf.string, tf.int32, tf.int32],
                                    shapes=[(3,), (3,), (3,)],
                                    shared_name=None, name=None)
    enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')

    image_batch, label_batch = create_input_pipeline(input_queue, image_size, nrof_preprocess_threads,batch_size_placeholder)

    image_batch = tf.identity(image_batch, 'image_batch')
    image_batch = tf.identity(image_batch, 'input')
    label_batch = tf.identity(label_batch, 'label_batch')

    prelogits = face_net(image_batch, face_class, phase_train_placeholder)

    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, face_class]), 3, 1)
    tloss = triplet_loss(anchor, positive, negative, alfa)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    triplet_lossx = tf.add_n([tloss]+regularization_losses,name= 'triplet_lossx')
    train_op = trian_optimizer(triplet_lossx, global_step, 'ADAM', learning_rate, 0.9999, tf.global_variables(),
                    log_histograms=True)
    # train_op = trian_optimizer(triplet_lossx,learning_rate)


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

            step = sess.run(global_step, feed_dict=None)
            img_train = []
            lab_train = []
            for i in range(epoch_num):
                while True:
                    index_epoch = []
                    for i in range(5):
                        xxx = random.randint(0, len(lab_data) - 6)
                        index_epoch.append(xxx)
                        index_epoch.append(xxx + 1)
                        index_epoch.append(xxx + 2)
                        index_epoch.append(xxx + 3)
                        index_epoch.append(xxx + 4)
                        index_epoch.append(xxx + 5)
                    label_epoch = np.array(lab_data)[index_epoch]
                    label_name = {}
                    for alabel in label_epoch:
                        if str(alabel) not in label_name:
                            label_name[str(alabel)] = 1
                        else:
                            label_name[str(alabel)] += 1
                    max_name = max(label_name, key=label_name.get)
                    if label_name[max_name] > 1:
                        break
                image_epoch = np.array(img_data)[index_epoch]
                img_train.extend(image_epoch)
                lab_train.extend(label_epoch)

            # print('你说这是什么事:lab_train', type(lab_train), len(lab_train),lab_train[-10:])# , lab_train
            # print('你说这是什么事:img_train', type(img_train), len(img_train),img_train[-10:]) # , img_train
            labels_array = np.reshape(np.expand_dims(np.array(lab_train), 1),(-1,3))
            image_paths_array = np.reshape(np.expand_dims(np.array(img_train), 1),(-1,3))
            # print('labels_array', labels_array)
            # print('image_paths_array', image_paths_array)
            control_value =0
            control_array = np.ones_like(labels_array) * control_value
            control_array = np.reshape(control_array,(-1,3))

            # print('control_array',control_array.shape)
            # print('image_paths_array',image_paths_array.shape)
            # print('labels_array',labels_array.shape)

            # print('control_array', control_array)
            sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array,
                                  control_placeholder: control_array})

            # print('我日有没有执行到')

            emb_arry = np.zeros((len(lab_train),face_class),dtype=float)
            batch_number =0
            while batch_number < epoch_num:
                # tensor_list = [total_loss, train_op,  regularization_losses, prelogits, cross_entropy_mean,prelogits_norm, accuracy, prelogits_center_loss]
                tensor_list = [label_batch,prelogits]
                feed_dict = {phase_train_placeholder:True, batch_size_placeholder:batch_size}
                label_into_ ,prelogits_= sess.run(tensor_list , feed_dict=feed_dict)

                emb_arry[batch_number*batch_size:batch_number*batch_size+batch_size]=prelogits_
                # print('label_into_',label_into_.shape, label_into_)
                # print('prelogits_',prelogits_.shape, prelogits_)
                batch_number +=1


            # print('label_into_',label_into_.shape, label_into_)
            # print('完成',emb_arry.shape,emb_arry.flatten().shape,emb_arry[-10:])
            trip_image_arr,trip_label_arr = get_apn_image(emb_arry.flatten(), img_train, lab_train,batch_size,epoch_num)

            nrof_batches = int(epoch_num/batch_size)
            triplet_labels_array = np.reshape(np.arange(epoch_num*3), (-1, 3))
            triplet_paths_array = np.reshape(np.expand_dims(np.array(trip_image_arr), 1), (-1, 3))
            control_value = 0
            triplet_control_array = np.ones_like(triplet_labels_array) * control_value
            triplet_control_array =np.reshape( triplet_control_array, (-1, 3))
            # print('control_array',triplet_control_array.shape,triplet_control_array[-10:])
            # print('triplet_paths_array',triplet_paths_array.shape, triplet_paths_array[-10:])
            # print('triplet_labels_array',triplet_labels_array.shape, triplet_labels_array[-10:])

            sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: triplet_labels_array,control_placeholder: triplet_control_array})
            i=0
            while i < nrof_batches:
                feed_dict = {batch_size_placeholder: batch_size,phase_train_placeholder: True}
                loss_, _,  emb, lab = sess.run([triplet_lossx, train_op, embeddings, label_batch],feed_dict=feed_dict)
                print('大批次',epoch,'\t小批次',i,'\t当前损失值',loss_)
                i+=1


            # print('看看筛选出来的三元组',len(trip_image_arr),trip_image_arr)
            # print('看看筛选出来的三元组',len(trip_label_arr), trip_label_arr)

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
# img_data,lab_data
nclass =int(lab_data[-1]+1)
batch_size =30
image_size = (160, 160)
nrof_preprocess_threads = 4
epoch_num = 600
face_class = 128
alfa = 0.3
max_epoch = 5000
learning_rate=0.0005
train_face_net(img_data,lab_data,nclass,image_size,nrof_preprocess_threads,face_class,alfa,max_epoch,batch_size,epoch_num,learning_rate)
