import tensorflow as tf
import numpy as np
import  os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

a = tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32)
with tf.Session() as sess:
    print(sess.run(tf.norm(a,axis=1)))