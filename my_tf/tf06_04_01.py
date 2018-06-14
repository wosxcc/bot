import numpy as np
import tensorflow as tf
x=tf.Variable(0,dtype=tf.float32)
y=pow(x,1)+20*x-123
learning=0.01
optimizer =tf.train.GradientDescentOptimizer(learning).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        sess.run(optimizer)
    print(sess.run(x))