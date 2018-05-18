import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (14,4)

n_observations = 200
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.2, 0.2, n_observations)
plt.scatter(xs, ys)
plt.show()


X= tf.placeholder(tf.float32,name='X')
Y=tf.placeholder(tf.float32,name='Y')

w=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

Y_pred=tf.add(tf.multiply(X,w),b)

w2 = tf.Variable(tf.random_normal([1],name='weight2'))
y_pred=tf.add(tf.multiply(tf.pow(X,2),w2),Y_pred)

w3=tf.Variable(tf.random_normal([1],name='weight3'))
Y_pred=tf.add(tf.multiply(tf.pow(X,3),w3),y_pred)

##定义loss函数
sample_num=xs.shape[0]
loss=tf.reduce_sum(tf.pow(Y_pred-Y,2))/sample_num ##求平均方差

###定义优化函数
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter('./data',sess.graph)

    for i in range(2001):
        init_loss = 0
        for x,y in zip(xs,ys):
            _,l=sess.run([optimizer,loss],feed_dict={X:x,Y:y})
        init_loss+=l

        if i %200==0:
            print('第{}次:'.format(i),l,sess.run([w,w2,w3,b]))
    writer.close()
    w,w2,w3,b=sess.run([w,w2,w3,b])
plt.plot(xs, ys, 'bo', label='Real data')    #真实值的散点
plt.plot(xs, xs*w + np.power(xs,2)*w2 + np.power(xs,3)*w3 + b, 'r', label='Predicted data')    #预测值的拟合线条
plt.legend()     #用于显示图例
plt.show()