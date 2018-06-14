import tensorflow as tf
import numpy as np

# #L2正则化
def get_weights(shape, lambd):

    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))    #把正则化加入集合losses里面
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
layer_dimension = [2, 10, 10, 10, 1]    #神经网络层节点的个数
n_layers = len(layer_dimension)         #神经网络的层数
cur_lay = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    out_dimension = layer_dimension[i]

    print(in_dimension, out_dimension)
    weights = get_weights([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_lay = tf.nn.relu(tf.matmul(cur_lay, weights)+bias)
    in_dimension = layer_dimension[i]

mess_loss = tf.reduce_mean(tf.square(y_-cur_lay))    #求平方和
tf.add_to_collection('losses', mess_loss)           #把均方误差也加入到集合里
loss = tf.add_n(tf.get_collection('losses'))
#tf.get_collection返回一个列表,内容是这个集合的所有元素
#add_n()把输入按照元素相加