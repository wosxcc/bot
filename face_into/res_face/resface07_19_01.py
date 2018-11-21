# coding:utf-8

import tensorflow as tf
import numpy as np
import cv2 as cv
import  os
from tensorflow.python.framework import  graph_util

import collections

label_lines = []
image_lines = []

blocks =[{'name': 'fnet1', 'net': [(64, 16, 1), (64, 16, 1), (64, 16, 2)]},
    {'name': 'fnet2', 'net': [(128, 32, 1), (128, 32, 1), (128, 32, 1), (128, 32, 2)]},
    # {'name': 'fnet3', 'net': [(256, 64, 1), (256, 64, 1), (256, 64, 1), (256, 64, 1), (256, 64, 1), (256, 64, 2)]},
    # {'name': 'fnet4', 'net': [(512, 256, 1), (512, 256, 1), (512, 256, 2)]}
    {'name': 'fnet4', 'net': [(256, 64, 1), (256, 64, 1), (256, 64, 2)]}
]



slim = tf.contrib.slim

def read_img(txt_name):
    label_lines = []
    image_lines = []
    txt_open = open(txt_name)
    txt_read = txt_open.read()
    txt_lines = txt_read.split('\n')

    for line in txt_lines:
        xlabel = []
        if len(line)>3:
            line_list = line.split(' ')
            image_lines.append(cv.imread(line_list[0]))
            xlabel.append(line_list[1])
            xlabel.append(line_list[2])
            for x in range(14):
                xlabel.append(line_list[117 + 2 + x * 2])
                xlabel.append(line_list[117 + 2 + x * 2 + 1])
            label_lines.append(xlabel)

            # label_lines.append(line_list[1:])

    label_linesc=[[float(i) for i in xline] for xline in label_lines]
    ximage_lines=np.array(image_lines, dtype='float32')
    ximage_lines/=255

    xlabel_linesc=np.array(label_linesc, dtype='float32')
    return ximage_lines,xlabel_linesc
def conv2d_same(inp_data,output_num, kernel_size,stride,scope=None):
    if stride ==1:
        return slim.conv2d(inp_data,output_num,kernel_size,stride,padding='SAME',scope=scope)

    else:
        pad_tatal = kernel_size -1
        pad_beg = pad_tatal//2              # 矩阵上边和左边填充数量
        pad_end = pad_tatal-pad_beg         # 矩阵下边和右边填充数量
        print('kernel_size',kernel_size,'pad_tatal',pad_tatal,'pad_beg',pad_beg,'pad_end',pad_end)
        inp_data = tf.pad(inp_data,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])   # 外部填充，使卷积能正常进行

        # tf.pad(       填充作用
        #     tensor, 是要填充的张量
        #     paddings, 也是一个张量，代表每一维填充多少行/列，但是有一个要求它的rank一定要和tensor的rank是一样的
        #     mode='CONSTANT', 可以取三个值，分别是"CONSTANT" ,"REFLECT","SYMMETRIC"
        #     name=None
        # )
        print('kernel_size,stride',kernel_size,stride)
        return  slim.conv2d(inp_data,output_num,kernel_size,stride=stride,padding='VALID', scope=scope)
        #padding='VALID' 不会超出屏幕外部，得到比原先平面小的平面



def subsample(inp_data,stride,scope = None):
    if stride == 1:
        return inp_data
    else:
        return slim.max_pool2d(inp_data,[1,1],stride=stride,scope=scope)


#               输入数据  128   32                步长
def bottleneck(inp_data,depth,depth_bottlenexk,stride,outputs_collections =None,scope=None):
    #  bottleneck残差学习单元,这是ResNet V2论文中提到的Full Preactivation Residual Unit的
    #     一个变种, 它和V1中的残差学习单元的主要区别有两点:
    #         1. 在每一层前都用了Batch Normalization
    #         2. 对输入进行preactivation，而不是在卷积进行激活函数处理
    print('depth,depth_bottlenexk',depth,depth_bottlenexk)
    with tf.variable_scope(scope, 'bottleneck_v2',[inp_data]) as sc:
        # depth_in = slim.utils.last_dimension(inp_data.get_shape(),min_rank=4)
        depth_in = slim.utils.last_dimension(inp_data.get_shape(), min_rank=4) ##shape的取第四个值
        # preact = slim.batch_norm(inp_data, activation_fn=tf.nn.relu, scope='preact')
        preact = slim.batch_norm(inp_data, activation_fn=tf.nn.relu,scope ='preact')   # 归一化（BN）加激活函数
        print('传入特征数量',depth,'数据特征数量',depth_in)

        if depth == depth_in:  #比较两个层数据结构
            shortcut = subsample(inp_data,stride,'shortcut')
        else:
            shortcut= slim.conv2d(preact,depth,[1,1],stride=stride,normalizer_fn =None,activation_fn =None,scope ='shortcut')
            print('这是进入了else',shortcut)
        print('天才第一步',preact)


        # 下面过程为：压缩0.25倍”→“卷积提特征”→“扩张”（是ResNet的特点，而MobileNetV2则是Inverted residuals,即：“扩张”→“卷积提特征”→ “压缩”）
        residual = slim.conv2d(preact,depth_bottlenexk,[1,1],stride=1,scope='conv1')    # 1*1的卷积核步长为1从原先的特征值到0.25倍要提取的特征值
        print('天才第二步', residual)
        residual = conv2d_same(residual,depth_bottlenexk,3,stride,scope='conv2')        # 3*3的卷积核步长为stride根据步长情况改变大小 ，抽取特征值
        print('天才第三步', residual)
        residual = slim.conv2d(residual,depth,[1,1],stride=1,normalizer_fn=None,activation_fn =None,scope='conv3')  # 用1*1的卷积核，对原网络抽取特征进行扩张
        print('天才第四步',residual)
        output= shortcut + residual #把前面层和当前层结果相加
        print('\n\n')
        return slim.utils.collect_named_outputs(outputs_collections,sc.name,output)

#                                                   全局池化,           重用
def res_net(inp_data,blocks, BATCH_SIZE,nclass,global_pool = True, include_root_block = True, reuse = None):
    with tf.variable_scope('res_net','face',[inp_data]) as xscope:
        end_points_collection = xscope.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d],activation_fn = None ,normalizer_fn =None):
            net = inp_data
        if include_root_block:                                                            # #做卷积方便后面对接
            with slim.arg_scope([slim.conv2d],activation_fn = None,normalizer_fn = None):
                net =conv2d_same(net,64,7,stride=1 ,scope='first')


        for block in blocks :
            with tf.variable_scope(block['name'],'block',[net]) as wxscope:
                i= 0
                for unit in block['net']:
                    i+=1
                    with tf.variable_scope('unit_%d'%(i+1),values=[net]):
                        unit_depth,unit_depth_bottleneck,unit_stride = unit
                        net = bottleneck(net,depth=unit_depth ,depth_bottlenexk=unit_depth_bottleneck,stride= unit_stride)
                        net = slim.utils.collect_named_outputs(None,wxscope.name,net)


        net =slim.batch_norm(net, activation_fn=tf.nn.relu,scope= 'postnorm')
        if global_pool:
            net = tf.reduce_mean(net,[1,2],name = 'pool5',keep_dims=True)# 求平均值
        if nclass is not None:
            net = slim.conv2d(net,nclass,[1,1],activation_fn = None,normalizer_fn=None,scope='logits')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            end_points['predictions']=slim.softmax(net,scope='predictions')
            net = tf.reshape(net,shape=[BATCH_SIZE,-1],name='output')

            return net, end_points



def losses(logits,labels):
    loss = tf.sqrt(tf.reduce_mean(tf.square(logits - labels)))
    return  loss

def optimizer_net(loss , learning_rate):
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0,name='global_step',trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(loss,global_step=global_step)
        return optimizer


def train_net():
    X_data ,Y_data = read_img(txt_name)
    max_size= len(Y_data)//BATCH_SIZE-1
    inp_x = tf.placeholder(tf.float32,shape=[None, IMG_W, IMG_H, 3],name='input')
    inp_y = tf.placeholder(tf.float32, shape=[None, Nclass],name='labels')

    net, end_points =res_net(inp_x,blocks,BATCH_SIZE,Nclass)
    train_loss = losses(net,inp_y)
    train_op = optimizer_net(train_loss,learning_rate)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(logs_train_dir)

    y_step=0
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path
        y_step= int(float(global_step.split('-')[1]))
        saver.restore(sess,ckpt.model_checkpoint_path)


    for step in range(MAX_STEP):
        xb = step%max_size
        input_x = X_data[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]
        # print(input_x.shape)
        _, xloss = sess.run([train_op,train_loss],feed_dict={
            inp_x: np.reshape(X_data[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE],(BATCH_SIZE,IMG_W,IMG_H,3)),
            inp_y: np.reshape(Y_data[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE],(BATCH_SIZE,Nclass))
        })

        if step % 50== 0:
            print('第%d个批次，当前loss值是%.5f'%(step+y_step,xloss))
        if step % 100 == 0:
            constant_graph = graph_util.convert_variables_to_constants(sess,sess.graph_def,
                                                                       ['res_net/output'])
            with tf.gfile.FastGFile(logs_train_dir+'face72.pb',mode='wb') as f :
                f.write(constant_graph.SerializeToString())

        if step %200 == 0 or (step + 1) ==MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
            saver.save(sess,checkpoint_path,global_step=step+y_step)
    sess.close()






learning_rate = 0.00001
Nclass = 30
IMG_W = 160
txt_name = 'trainc.txt'
IMG_H = 160
BATCH_SIZE = 10
MAX_STEP = 100000

logs_train_dir = './facem/log/'
train_net()

















