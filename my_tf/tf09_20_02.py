import cv2 as cv
import tensorflow as tf
import numpy as np
import  os

from tensorflow.python.ops import control_flow_ops
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

img = cv.imread('../my_cv/01.jpg')
img=np.array(img,dtype='float32')
img = img.transpose([2,0,1])
imgss =tf.concat(values=[img],axis=3)



def batch_norm(x, phase_train):  # pylint: disable=unused-variable
    """
    卷积映射的批量归一化。
    Args:
        x:           张量，4D BHWD输入映射
        n_out:       整数，输入深度图
        phase_train: 布尔tf.Value，TRUE指示训练阶段
        scope:       变量范围字符串
        affn:     是否转换输出
    Return:
        normed:     批量归一化映射
    Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
    """
    name = 'batch_norm'
    with tf.variable_scope(name):
        phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
        n_out = int(x.get_shape()[-1])

        print('nout',n_out)
        beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                           name=name + '/beta', trainable=True, dtype=x.dtype)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                            name=name + '/gamma', trainable=True, dtype=x.dtype)

        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        print('batch_mean',batch_mean, 'batch_var',batch_var)
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


with tf.Session() as sess:


    print(sess.run(imgss))
    img_nrm =batch_norm(imgss, False)

    print(sess.run(img_nrm))

cv.imshow('img', img)
cv.waitKey()