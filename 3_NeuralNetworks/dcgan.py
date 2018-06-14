# -*- coding:utf-8 -*-

""" Deep Convolutional Generative Adversarial Network (DCGAN).

利用深度卷积生成对抗网络（DCGAN）生成
来自噪声分布的数字图像。
参考文献：
深度卷积生成的无监督表示学习
对抗网络一个Radford，L Mez，ScTala。阿西夫：1511.06434。
Links:
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434).
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

卷积神经网络在有监督学习中的各项任务上都有很好的表现，但在无监督学习领域，却比较少。本文介绍的算法将有监督学习中的CNN和无监督学习中的GAN结合到了一起。
在非CNN条件下，LAPGAN在图像分辨率提升领域也取得了好的效果。
与其将本文看成是CNN的扩展，不如将其看成GAN的扩展到CNN领域。而GAN的基本算法，可以参考对抗神经网络。

GAN无需特定的cost function的优势和学习过程可以学习到很好的特征表示，但是GAN训练起来非常不稳定，经常会使得生成器产生没有意义的输出。而论文的贡献就在于：

为CNN的网络拓扑结构设置了一系列的限制来使得它可以稳定的训练。
使用得到的特征表示来进行图像分类，得到比较好的效果来验证生成的图像特征表示的表达能力
对GAN学习到的filter进行了定性的分析。
展示了生成的特征表示的向量计算特性。

，以图像生成模型举例。假设我们有一个图片生成模型（generator），它的目标是生成一张真实的图片。与此同时我们有一个图像判别模型（discriminator），它的目标是能够正确判别一张图片是生成出来的还是真实存在的。那么如果我们把刚才的场景映射成图片生成模型和判别模型之间的博弈，就变成了如下模式：生成模型生成一些图片->判别模型学习区分生成的图片和真实图片->生成模型根据判别模型改进自己，生成新的图片->····
"""



from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import  os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print('看看值',mnist)
# Training Params
num_steps = 10000
batch_size = 32

# Network Params
image_dim = 784 # 28*28 pixels * 1 channel
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200 # Noise data points


# 创造的网络
#输入图像的噪声，输出
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.layers.dense(x, units=6 * 6 * 128)
        x = tf.nn.tanh(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)    # tf.layers.conv2d_transpose 解卷积
        # 卷积，图像形状：（batch，28, 28, 1）
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        #应用SigMID来剪辑0到1之间的值。
        x = tf.nn.sigmoid(x)
        print(x)
        return x


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
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
    return x

# 建立网络
# 网络输入
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

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
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.001)

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
    with tf.Session() as sess:

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

            # Prepare Input Data
            # 获取下一批MNIST数据（仅需要图像，而不是标签）
            batch_x, _ = mnist.train.next_batch(batch_size)
            # print(batch_x.shape)
            batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
            # print(batch_x.shape)
            # 生成噪声馈送到生产者
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

            # Prepare Targets (Real image: 1, Fake image: 0)
            # The first half of data fed to the generator are real images,
            # the other half are fake images (coming from the generator).
            batch_disc_y = np.concatenate(
                [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
            # Generator tries to fool the discriminator, thus targets are 1.
            batch_gen_y = np.ones([batch_size])

            # Training
            feed_dict = {real_image_input: batch_x, noise_input: z,
                         disc_target: batch_disc_y, gen_target: batch_gen_y}
            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                    feed_dict=feed_dict)
            if i % 100 == 0 or i == 1:
                print('Step %i: 产生图像 Loss值: %f, 对比 Loss值: %f' % (i, gl, dl))
            if i % 500 == 0 or i == num_steps:
                checkpoint_path = os.path.join(updata_gan,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=i)

        # Generate images from noise, using the generator network.

            ##使用生成器网络从噪声生成图像
            # if  i % 400 == 0:
        f, a = plt.subplots(5, 10, figsize=(10, 5))
        for i in range(10):
            # Noise input.
            z = np.random.uniform(-1., 1., size=[5, noise_dim])

            g = sess.run(gen_sample, feed_dict={noise_input: z})

            for j in range(5):
                # 从噪声中生成图像。扩展到Matlab图形的3个通道。
                img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                                 newshape=(28, 28, 3))
                a[j][i].imshow(img)

        f.show()
        plt.draw()
        plt.waitforbuttonpress(30)


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
                z = np.random.uniform(-0.5, 0.5, size=[5, noise_dim])
                g = sess.run(gen_sample, feed_dict={noise_input: z})
                for j in range(5):
                    # 从噪声中生成图像。扩展到Matlab图形的3个通道。
                    img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                                     newshape=(28, 28, 3))
                    a[j][i].imshow(img)
            f.show()
            plt.draw()
            plt.show()
gangangan()