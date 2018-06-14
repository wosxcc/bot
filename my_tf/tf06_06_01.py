import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

image_raw_data = tf.gfile.FastGFile('.//image//1.jpg','rb').read()

with tf.Session() as sess:
     img_data = tf.image.decode_jpeg(image_raw_data)
     plt.imshow(img_data.eval())
     plt.show()

     # 上下翻转
     flipped1 = tf.image.flip_up_down(img_data)
     plt.imshow(flipped1.eval())
     plt.show()
     # 左右翻转
     flipped2 = tf.image.flip_left_right(img_data)
     plt.imshow(flipped2.eval())
     plt.show()
     #对角线翻转
     transposed = tf.image.transpose_image(img_data)
     plt.imshow(transposed.eval())
     plt.show()

     # 以一定概率上下翻转图片。
     #flipped = tf.image.random_flip_up_down(img_data)
     # 以一定概率左右翻转图片。
     #flipped = tf.image.random_flip_left_right(img_data)