import tensorflow as tf
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

aa = tf.get_variable('centers', [5,8], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
lables = tf.constant([4,1,3,4,0,3,4,2])
centers_batch = tf.gather(aa, lables)
with tf.Session() as sess:
    init_g = tf.global_variables_initializer()
    sess.run(init_g)
    print(sess.run(aa))
    print(sess.run(centers_batch))
    # print(sess.run(tf.gather(aa, lables)))