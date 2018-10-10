import tensorflow as tf

# a=[[1,3,4,5,6],
#    [9,87,5,347,2]]
a =tf.constant([[1,-3,4,5,6],
   [9,87,5,347,2]])
with tf.Session() as sess:

    # xa = tf.reshape(a,[None])
    print(sess.run(tf.reduce_max(a)))
    print(sess.run(tf.reduce_min(a)))
    # print(sess.run(tf.arg_max(),0))



# python src/align/align_dataset_mtcnn.py ~/data/raw/ ~/data/casia_maxpy_mtcnnpy_182 --image_size 182 --margin 44