import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

a = [0, 1, 1, 2, 2, 10]

sess = tf.Session()
print(sess.run(tf.bincount(a)))