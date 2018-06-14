import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

n_observations = 500
datax = np.linspace(-3, 3, n_observations)
# datax = np.arange(-50,50,0.5,dtype=float)
print(len(datax))
datay=2*np.power(datax,2) + 3*datax+ np.random.uniform(0.8,1.8,n_observations)
# print(datay)

# plt.scatter(datax,datay,s=2,c='r')
# plt.plot(datax, datax*3 + np.power(datax,2)*2 + 1.3, 'g', label='Predicted data')    #预测值的拟合线条
#
# # plt.plot(datax,datay=2*np.power(datax,2) + 3*datax+ np.random.uniform(-0.6,0.6,n_observations),)
# plt.show()

def get_weights(shape,lamdb):
    var= tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamdb)(var))

    return var

int_X=tf.placeholder(dtype=tf.float32,name='X')
int_Y=tf.placeholder(dtype=tf.float32,name='Y')


w1=tf.Variable(tf.random_normal([1]),dtype=tf.float32 ,name='weight')
w2=tf.Variable(tf.random_normal([1]),dtype=tf.float32 ,name='weight2')
b=tf.Variable(tf.random_normal([1]),dtype=tf.float32 ,name='bias')


Y_guji=tf.add(tf.multiply(w1,tf.pow(int_X,2)),tf.multiply(w2,int_X))
Y_guji=tf.add(Y_guji,b)
sman=datax.shape[0]

layer_count=[2,1]
cur_lay=int_X
int_dimension=layer_count[0]
for i in range(1,len(layer_count)):
    out_dimension=layer_count[i]
    weight=get_weights([int_dimension,out_dimension],0.001)
    bias =tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    cur_lay=tf.nn.relu(tf.matmul(cur_lay,weight)+bias)
    int_dimension=layer_count[i]
mess_loss=tf.reduce_mean(tf.square(Y_guji-int_Y))
tf.add_to_collection('losses',mess_loss)
loss= tf.add_n(tf.get_collection('losses'))

# loss=tf.reduce_sum(tf.pow(Y_guji-int_Y,2))/sman

optimize=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    for i  in range(5000):
        sess.run([optimize,loss],feed_dict={int_X:datax,int_Y:datay})

    w1,w2,b=sess.run([w1, w2, b])
    print('w1,w2,b',w1,w2,b)
