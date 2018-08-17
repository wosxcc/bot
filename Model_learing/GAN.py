# -*- coding:utf-8 -*-

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import  os
import  cv2 as cv


# Training Params
num_steps = 100000
batch_size = 20

# Network Params
image_dim = 27648 # 28*28 pixels * 1 channel
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200 # Noise data points
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_batch(image, img_W, img_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    input_queue = tf.train.slice_input_producer([image])
    image = tf.image.decode_jpeg(input_queue[0], channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, img_W, img_H)
    image = tf.image.per_image_standardization(image)
    image_batch = tf.train.batch([image], batch_size=batch_size,  num_threads=32, capacity=capacity)

    image_batch = tf.cast(image_batch, tf.float32)
    print(image_batch)
    return image_batch
path_file = 'E:/xbot/my_tf/gangan'
img_name=[]
for filess in  os.listdir(path_file):
    img_name.append(path_file+'/'+filess)
print(img_name)
num_batch = len(img_name)//batch_size

def get_nextbatch(index):
    ximage_batch = []
    images = img_name[index * batch_size:(index + 1) * batch_size]
    for img in images:
        arr = cv.imread(img)
        arr = cv.resize(arr,(96,96),cv.INTER_CUBIC)
        arr = np.array(arr)
        arr =arr.astype('float32')/127.5-1
        ximage_batch.append(arr)
    return ximage_batch



# 创造的网络
#输入图像的噪声，输出
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.layers.dense(x, units=23 * 23 * 128)
        x = tf.nn.tanh(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 23, 23, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)    # tf.layers.conv2d_transpose 解卷积
        # 卷积，图像形状：（batch，96, 96, 3）
        x = tf.layers.conv2d_transpose(x, 3, 2, strides=2)
        #应用SigMID来剪辑0到1之间的值。
        x = tf.nn.sigmoid(x)

        return x


# def deconv2d(input_, output_shape,
#          k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#          name="deconv2d", with_w=False):
#     with tf.variable_scope(name):
#         # filter : [height, width, output_channels, in_channels]
#         w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],initializer=tf.random_normal_initializer(stddev=stddev))
#         try:
#             deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
#         # Support for verisons of TensorFlow before 0.7.0
#         except AttributeError:
#             deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
#         biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#         deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
#         if with_w:
#             return deconv, w, biases
#         else:
#             return deconv
#
# def generator(self, z, y=None):
#     with tf.variable_scope("generator") as scope:
#         if not self.y_dim:
#             # s是输出图片的大小，比如s是64，s2为32，s4为16，s8为8,s16为4
#             s = self.output_size
#             s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
#             # project `z` and reshape
#             self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s16 * s16, 'g_h0_lin', with_w=True)
#             self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
#             h0 = tf.nn.relu(self.g_bn0(self.h0))
#             self.h1, self.h1_w, self.h1_b = deconv2d(h0,[self.batch_size, s8, s8, self.gf_dim * 4], name='g_h1',with_w=True)
#             h1 = tf.nn.relu(self.g_bn1(self.h1))
#             h2, self.h2_w, self.h2_b = deconv2d(h1,[self.batch_size, s4, s4, self.gf_dim * 2], name='g_h2',with_w=True)
#             h2 = tf.nn.relu(self.g_bn2(h2))
#             h3, self.h3_w, self.h3_b = deconv2d(h2,[self.batch_size, s2, s2, self.gf_dim * 1], name='g_h3',with_w=True)
#             h3 = tf.nn.relu(self.g_bn3(h3))
#             h4, self.h4_w, self.h4_b = deconv2d(h3,[self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)
#             return tf.nn.tanh(h4)
#         else:
#             s = self.output_size
#             s2, s4 = int(s / 2), int(s / 4)
#             # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
#             yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
#             z = tf.concat(1, [z, y])
#             h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
#             h0 = tf.concat(1, [h0, y])
#             h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim * 2 * s4 * s4, 'g_h1_lin')))
#             h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
#             h1 = conv_cond_concat(h1, yb)
#             h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2')))
#             h2 = conv_cond_concat(h2, yb)
#         return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))


# 网络#鉴频器
#图像输入/输出：伪造图像的实时预测
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(x, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        # x = tf.layers.conv2d(x, 256, 5)
        # x = tf.nn.tanh(x)
        # x = tf.layers.average_pooling2d(x, 2, 2)
        # x = tf.layers.conv2d(x, 128, 5)
        # x = tf.nn.tanh(x)
        # x = tf.layers.average_pooling2d(x, 2, 2)
        # x = tf.layers.conv2d(x, 256, 5)
        # x = tf.nn.tanh(x)
        # x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
    return x

# 建立网络
# 网络输入
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 96, 96,3])
# Build Generator Network
gen_sample = generator(noise_input)
# 建立2个鉴别器网络（一个来自噪声输入，一个来自生成的样本）
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real, disc_fake], axis=0)

#建立叠层发创造/鉴别器
stacked_gan = discriminator(gen_sample, reuse=True)

# 建立目标（真实或虚假的图像）
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

#建造损失
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=gen_target))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.0001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.0001)

# 每个优化器的训练变量
#在TensorFlow默认情况下，所有变量都由每个优化器更新，所以我们
#需要精确地对它们中的每一个变量进行更新。
#创造网络变量
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# 鉴别器网络变量
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# 创建训练优化
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
def gan_train():
    mygan = get_batch(img_name, 96, 96, batch_size, batch_size)

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Run the initializer
        updata_gan = './gan/log'
        train_write= tf.summary.FileWriter(updata_gan,sess.graph)
        saver= tf.train.Saver()
        sess.run(init)
        ckpt=tf.train.get_checkpoint_state(updata_gan)
        if ckpt and ckpt.model_checkpoint_path:
            global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess,ckpt.model_checkpoint_path)
        for i in range(1, num_steps+1):
            # 生成噪声馈送到生产者
            for j in range(5):
                z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
                batch_disc_y = np.concatenate(
                    [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
                batch_gen_y = np.ones([batch_size])
                # Training
                feed_dict = {real_image_input: get_nextbatch(j), noise_input: z,
                             disc_target: batch_disc_y, gen_target: batch_gen_y}
                _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                        feed_dict=feed_dict)


            if i % 20 == 0 or i == 1:
                print('Step %i: 产生图像 Loss值: %f, 对比 Loss值: %f' % (i, gl, dl))
            if i % 100 == 0 or i == num_steps:
                checkpoint_path = os.path.join(updata_gan,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=i)

        # Generate images from noise, using the generator network.

            ##使用生成器网络从噪声生成图像
            # if  i % 400 == 0:
        f, a = plt.subplots(5, 10, figsize=(10, 5))
        for i in range(10):
            # Noise input.
            z = np.random.uniform(-1., 1., size=[4, noise_dim])
            g = sess.run(gen_sample, feed_dict={noise_input: z})
            for j in range(4):
                # 从噪声中生成图像。扩展到Matlab图形的3个通道。
                img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                                 newshape=(96, 96, 3))
                a[j][i].imshow(img)
        f.show()
        plt.draw()
        plt.waitforbuttonpress(20)

def gangangan():
    updata_gan = './gan/log'
    # with tf.Graph().as_default():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(updata_gan)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            f, a = plt.subplots(5, 10, figsize=(10, 5))
            for i in range(10):
                # Noise input.
                z = np.random.uniform(-1., 1., size=[5, noise_dim])
                images = sess.run(gen_sample, feed_dict={noise_input: z})
                # print(images)
                for j in range(5):
                    # 从噪声中生成图像。扩展到Matlab图形的3个通道。
                    image = images[j,:,:,:]
                    image += 1
                    image *=127.5
                    image = np.clip(image,0,255).astype(np.uint8)
                    image =np.reshape(image,(96, 96, -1))
                    # cv.imshow(str(i)+str(j),image)
                    # img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                    #                  newshape=(96, 96, 3))
                    a[j][i].imshow(image)
            f.show()
            plt.draw()
            plt.show()
# gangangan()
gan_train()