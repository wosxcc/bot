import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('.//image//1.jpg','rb').read()

with tf.Session() as sess:
     img_data = tf.image.decode_jpeg(image_raw_data)
     plt.imshow(img_data.eval())
     plt.show()

     # 将图片的亮度-0.5。
     adjusted = tf.image.adjust_brightness(img_data, -0.5)
     plt.imshow(adjusted.eval())
     plt.show()

     # 将图片的亮度0.5
     adjusted = tf.image.adjust_brightness(img_data, 0.5)
     plt.imshow(adjusted.eval())
     plt.show()
     # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
     adjusted = tf.image.random_brightness(img_data, max_delta=0.5)
     plt.imshow(adjusted.eval())
     plt.show()
     # 将图片的对比度-5
     adjusted = tf.image.adjust_contrast(img_data, -5)
     plt.imshow(adjusted.eval())
     plt.show()
     # 将图片的对比度+5
     adjusted = tf.image.adjust_contrast(img_data, 5)
     plt.imshow(adjusted.eval())
     plt.show()
     # 在[lower, upper]的范围随机调整图的对比度。
     adjusted = tf.image.random_contrast(img_data, 0.1, 0.6)
     plt.imshow(adjusted.eval())
     plt.show()

     #调整图片的色相
     adjusted = tf.image.adjust_hue(img_data, 0.1)
     plt.imshow(adjusted.eval())
     plt.show()

     # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
     adjusted = tf.image.random_hue(img_data, 0.5)
     plt.imshow(adjusted.eval())
     plt.show()


     # 将图片的饱和度-5。
     adjusted = tf.image.adjust_saturation(img_data, -5)
     plt.imshow(adjusted.eval())
     plt.show()


     # 在[lower, upper]的范围随机调整图的饱和度。
     adjusted = tf.image.random_saturation(img_data, 0, 5)

     # 归一化操作
     # 将代表一张图片的三维矩阵中的数字均值变为0，方差变为1。
     adjusted = tf.image.per_image_standardization(img_data)