import tensorflow as tf
import  numpy as np
import  matplotlib.pyplot as plt
x_date=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_date.shape)
y_date=np.square(x_date)+noise

x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])


Weights_L1=tf.Variable(tf.random_normal([1,10]))
biases_L1=tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
L1=tf.nn.tanh(Wx_plus_b_L1,name='xcc1')


Weights_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction= tf.nn.tanh(Wx_plus_b_L2)

loss=tf.reduce_mean(tf.square(y-prediction))

train_set=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.Session() as  sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(2000):
        sess.run(train_set,feed_dict={x:x_date,y:y_date})
        prediction_value= sess.run(prediction,feed_dict={x:x_date})

    plt.figure()
    plt.scatter(x_date,y_date)
    plt.plot(x_date,prediction_value,'r-',lw=3)
    plt.show()