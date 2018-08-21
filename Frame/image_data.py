import cv2 as cv
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from Frame.to_batch_img import RoIDataLayer
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

try:
  import cPickle as pickle
except ImportError:
  import pickle
import os

net =vgg16(batch_size=1)
mdata =[]
num_classes =2
image_batch =1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
output_dir ='./output'
imgsp=np.zeros((1,480,480,3))
mpath ='E:/BOT_Person/trainb480'

# ckpt_path ='./output/vgg16_faster_rcnn_iter_35200.ckpt'
ckpt_path = 'E:/Model/Faster-RCNN/data/imagenet_weights/vgg_16.ckpt'


for file in os.listdir(mpath):
    if file[-3:] == 'jpg':
        imgae_data = {}
        img = cv.imread(mpath+'/'+file)

        img =cv.resize(img ,(480,480),interpolation=cv.INTER_CUBIC)
        txt_open = open(mpath+'/'+file[:-3]+'txt')
        txt_read = txt_open.read()
        gt_boxes = []
        for line in txt_read.split('\n'):
            if len(line)>5:
                bbox =  [float(i) for  i in line.split(' ')]
                nclass=  bbox[0]+1
                x1 = (bbox[1]-bbox[3]/2) * 480
                y1 = (bbox[2]-bbox[4]/2) * 480
                x2 = (bbox[1]+bbox[3]/2) * 480
                y2 = (bbox[2]+bbox[4]/2) * 480
                gt_boxes.append([x1,y1,x2,y2,nclass])
                # print(bbox)

        imgae_data['gt_boxes']=np.array(gt_boxes)
        imgae_data['im_info']=np.array([[480,480,1.0]])
        imgsp[0] =(img.astype(np.float32)-127.5)
        imgae_data['data']=imgsp

        # print(imgae_data['data'].shape)
        # print('比博客是', imgae_data['im_info'].shape)
        # print(imgae_data['gt_boxes'].shape, imgae_data['data'].shape)
        mdata.append(imgae_data)
        # print(mdata)
        # cv.imshow('img',img)
        # cv.waitKey()

data_layer = RoIDataLayer(mdata, num_classes)



def train():
    # Create session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # print('说好了的一起走。',tfconfig)
    sess = tf.Session(config=tfconfig)

    with sess.graph.as_default():

        tf.set_random_seed(3)
        # print('到期了',num_classes)
        layers = net.create_architecture(sess, "TRAIN", num_classes, tag='default')
        loss = layers['total_loss']
        lr = tf.Variable(0.001, trainable=False)
        momentum = 0.9
        optimizer = tf.train.MomentumOptimizer(lr, momentum)

        gvs = optimizer.compute_gradients(loss)

        # Double bias
        # Double the gradient of the bias if set
        if True:
            final_gvs = []
            with tf.variable_scope('Gradient_Mult'):
                for grad, var in gvs:
                    scale = 1.
                    if True and '/biases:' in var.name:
                        scale *= 2.
                    if not np.allclose(scale, 1.0):
                        grad = tf.multiply(grad, scale)
                    final_gvs.append((grad, var))
            train_op = optimizer.apply_gradients(final_gvs)
        else:
            train_op = optimizer.apply_gradients(gvs)
        msaver = tf.train.Saver(max_to_keep=100000)

    # Load weights
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(ckpt_path))
    variables = tf.global_variables()
    # Initialize all variables first
    sess.run(tf.variables_initializer(variables, name='init'))
    var_keep_dic = get_variables_in_checkpoint_file(ckpt_path)
    # Get the variables to restore, ignorizing the variables to fix
    # print('为什么会没有值',var_keep_dic)
    variables_to_restore = net.get_variables_to_restore(variables, var_keep_dic)

    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, ckpt_path)
    # Need to fix the variables before loading, so that the RGB weights are changed to BGR
    # For VGG16 it also changes the convolutional weights fc6 and fc7 to
    # fully connected weights
    net.fix_variables(sess,ckpt_path)
    sess.run(tf.assign(lr, 0.001))
    last_snapshot_iter = 0

    timer = Timer()
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()
    while iter < 40000 + 1:
        # Learning rate
        if iter == 5000 + 1:
            #                 学习率            学习率        *       降低学习率的因素
            sess.run(tf.assign(lr, 0.001 * 0.1))

        timer.tic()
        # Get training data, one batch at a time
        blobs = data_layer.forward()
        print('比博客是', blobs['im_info'].shape, blobs['gt_boxes'].shape, blobs['data'].shape)
        # blobs['im_info'] 图像大小和比例(img_H,img_W,缩放比例)   , blobs['gt_boxes']框的位置和类别（x1,y1,x2,y2,class）,blobs['data']是一张600*800*3的图像
        # Compute the graph without summary
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = net.train_step(sess, blobs, train_op)
        timer.toc()
        iter += 1

        # Display training information
        if iter % (10) == 0:
            print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                  '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n ' % \
                  (iter, 40000., total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box))
            print('speed: {:.3f}s / iter'.format(timer.average_time))

        if iter % 200 == 0:
            snapshot(sess, iter,msaver )    # 保存模型操作
def get_variables_in_checkpoint_file(file_name):
    # print('第一步还是对的',file_name)
    # try:
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    # print('到这步还是对的',file_name)

    var_to_shape_map = reader.get_variable_to_shape_map()
    return var_to_shape_map
    # except Exception as e:  # pylint: disable=broad-except
    #     print('出现错误')
    #     print(str(e))
    #     if "corrupted compressed block contents" in str(e):
    #         print("It's likely that your checkpoint file has been compressed "
    #               "with SNAPPY.")

def snapshot(sess, iter,msaver):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store the model snapshot
    filename = 'vgg16_faster_rcnn_iter_{:d}'.format(iter) + '.ckpt'
    filename = os.path.join(output_dir, filename)

    # print("看看你这个filename充钱就可以",filename)
    msaver.save(sess, filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    # Also store some meta information, random state, etc.
    nfilename = 'vgg16_faster_rcnn_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(output_dir, nfilename)
    # current state of numpy random
    st0 = np.random.get_state()
    # current position in the database
    cur = data_layer._cur
    # current shuffled indeces of the database
    perm = data_layer._perm

    # Dump the meta info
    with open(nfilename, 'wb') as fid:
        pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
        pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
        pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

    return filename, nfilename


train()