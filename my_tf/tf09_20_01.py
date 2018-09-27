import tensorflow as tf
import numpy as np


img = tf.Variable(tf.random_normal([9,9]))
print('img',img)
axis = list(range(len(img.get_shape())-1))

print('axis',axis)

mean,variance = tf.nn.moments(img,axis)
print('mean',mean)
print('variance',variance)



tf.norm
tf.nn.batch_norm_with_global_normalization
tf.nn.batch_normalization