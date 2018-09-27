import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
t = tf.truncated_normal_initializer(stddev=0.05, seed=1)
v = tf.get_variable('v', [1], initializer=t)

with tf.Session() as sess:
    for i in range(1, 10, 1):
        sess.run(tf.global_variables_initializer())
        print(sess.run(v))

[-0.08113182]
[0.06396971]
[0.13587774]
[0.05517125]
[-0.02088852]
[-0.03633211]
[-0.06759059]
[-0.14034753]
[-0.16338211]



[-0.04056591]
[0.03198485]
[0.06793887]
[0.02758562]
[-0.01044426]
[-0.01816605]
[-0.0337953]
[-0.07017376]
[-0.08169106]