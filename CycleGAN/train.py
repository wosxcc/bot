from __future__ import print_function

import argparse
from datetime import datetime
from random import shuffle
import random
import os
import sys
import time
import math
import tensorflow as tf
import numpy as np
import glob
import cv2

from CycleGAN.train_image_reader import *
from CycleGAN.net import *

parser = argparse.ArgumentParser(description='')

parser.add_argument("--snapshot_dir", default='./snapshots', help="path of snapshots")  # 保存模型的路径
parser.add_argument("--out_dir", default='./train_out', help="path of train outputs")  # 训练时保存可视化输出的路径
parser.add_argument("--image_size", type=int, default=256, help="load image size")  # 网络输入的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed")  # 随机数种子
parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate for adam')  # 基础学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')  # 训练的epoch数量
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100,
                    help='# of epoch to decay lr')  # 训练中保持学习率不变的epoch数量
parser.add_argument("--lamda", type=float, default=10.0, help="L1 lamda")  # 训练中L1_Loss前的乘数
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')  # adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=200,
                    help="times to summary.")  # 训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=100, help="times to write.")  # 训练中每过多少step保存可视化结果
parser.add_argument("--save_pred_every", type=int, default=10000, help="times to save.")  # 训练中每过多少step保存模型(可训练参数)
parser.add_argument("--x_train_data_path", default='./dataset/horse2zebra/trainA/',
                    help="path of x training datas.")  # x域的训练图片路径
parser.add_argument("--y_train_data_path", default='./dataset/horse2zebra/trainB/',
                    help="path of y training datas.")  # y域的训练图片路径

args = parser.parse_args()


def save(saver, sess, logdir, step):  # 保存模型的save函数
    model_name = 'model'  # 保存的模型名前缀
    checkpoint_path = os.path.join(logdir, model_name)  # 模型的保存路径与名称
    if not os.path.exists(logdir):  # 如果路径不存在即创建
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)  # 保存模型
    print('The checkpoint has been created.')


def cv_inv_proc(img):  # cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = (img + 1.) * 127.5
    return img_rgb.astype(np.float32)  # 返回bgr格式的图像，方便cv2写图像


def get_write_picture(x_image, y_image, fake_y, fake_x_, fake_x, fake_y_):  # get_write_picture函数得到训练过程中的可视化结果
    x_image = cv_inv_proc(x_image)  # 还原x域的图像
    y_image = cv_inv_proc(y_image)  # 还原y域的图像
    fake_y = cv_inv_proc(fake_y[0])  # 还原生成的y域的图像
    fake_x_ = cv_inv_proc(fake_x_[0])  # 还原重建的x域的图像
    fake_x = cv_inv_proc(fake_x[0])  # 还原生成的x域的图像
    fake_y_ = cv_inv_proc(fake_y_[0])  # 还原重建的y域的图像
    row1 = np.concatenate((x_image, fake_y, fake_x_), axis=1)  # 得到训练中可视化结果的第一行
    row2 = np.concatenate((y_image, fake_x, fake_y_), axis=1)  # 得到训练中可视化结果的第二行
    output = np.concatenate((row1, row2), axis=0)  # 得到训练中可视化结果
    return output


def make_train_data_list(x_data_path, y_data_path):  # make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    x_input_images_raw = glob.glob(os.path.join(x_data_path, "*"))  # 读取全部的x域图像路径名称列表
    y_input_images_raw = glob.glob(os.path.join(y_data_path, "*"))  # 读取全部的y域图像路径名称列表
    x_input_images, y_input_images = add_train_list(x_input_images_raw, y_input_images_raw)  # 将x域图像数量与y域图像数量对齐
    return x_input_images, y_input_images


def add_train_list(x_input_images_raw, y_input_images_raw):  # add_train_list函数将x域和y域的图像数量变成一致
    if len(x_input_images_raw) == len(y_input_images_raw):  # 如果x域和y域图像数量本来就一致，直接返回
        return shuffle(x_input_images_raw), shuffle(y_input_images_raw)
    elif len(x_input_images_raw) > len(y_input_images_raw):  # 如果x域的训练图像数量大于y域的训练图像数量，则随机选择y域的图像补充y域
        mul_num = int(len(x_input_images_raw) / len(y_input_images_raw))  # 计算两域图像数量相差的倍数
        y_append_num = len(x_input_images_raw) - len(y_input_images_raw) * mul_num  # 计算需要随机出的y域图像数量
        append_list = [random.randint(0, len(y_input_images_raw) - 1) for i in range(y_append_num)]  # 得到需要补充的y域图像下标
        y_append_images = []  # 初始化需要被补充的y域图像路径名称列表
        for a in append_list:
            y_append_images.append(y_input_images_raw[a])
        y_input_images = y_input_images_raw * mul_num + y_append_images  # 得到数量与x域一致的y域图像
        shuffle(x_input_images_raw)  # 随机打乱x域图像顺序
        shuffle(y_input_images)  # 随机打乱y域图像顺序
        return x_input_images_raw, y_input_images  # 返回数量一致的x域和y域图像路径名称列表
    else:  # 与elif中的逻辑一致，只是x与y互换，不再赘述
        mul_num = int(len(y_input_images_raw) / len(x_input_images_raw))
        x_append_num = len(y_input_images_raw) - len(x_input_images_raw) * mul_num
        append_list = [random.randint(0, len(x_input_images_raw) - 1) for i in range(x_append_num)]
        x_append_images = []
        for a in append_list:
            x_append_images.append(x_input_images_raw[a])
        x_input_images = x_input_images_raw * mul_num + x_append_images
        shuffle(y_input_images_raw)
        shuffle(x_input_images)
        return x_input_images, y_input_images_raw


def l1_loss(src, dst):  # 定义l1_loss
    return tf.reduce_mean(tf.abs(src - dst))


def gan_loss(src, dst):  # 定义gan_loss，在这里用了二范数
    return tf.reduce_mean((src - dst) ** 2)


def main():
    if not os.path.exists(args.snapshot_dir):  # 如果保存模型参数的文件夹不存在则创建
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir):  # 如果保存训练中可视化输出的文件夹不存在则创建
        os.makedirs(args.out_dir)
    x_datalists, y_datalists = make_train_data_list(args.x_train_data_path,
                                                    args.y_train_data_path)  # 得到数量相同的x域和y域图像路径名称列表
    tf.set_random_seed(args.random_seed)  # 初始一下随机数
    x_img = tf.placeholder(tf.float32, shape=[1, args.image_size, args.image_size, 3], name='x_img')  # 输入的x域图像
    y_img = tf.placeholder(tf.float32, shape=[1, args.image_size, args.image_size, 3], name='y_img')  # 输入的y域图像

    fake_y = generator(image=x_img, reuse=False, name='generator_x2y')  # 生成的y域图像
    fake_x_ = generator(image=fake_y, reuse=False, name='generator_y2x')  # 重建的x域图像
    fake_x = generator(image=y_img, reuse=True, name='generator_y2x')  # 生成的x域图像
    fake_y_ = generator(image=fake_x, reuse=True, name='generator_x2y')  # 重建的y域图像

    dy_fake = discriminator(image=fake_y, reuse=False, name='discriminator_y')  # 判别器返回的对生成的y域图像的判别结果
    dx_fake = discriminator(image=fake_x, reuse=False, name='discriminator_x')  # 判别器返回的对生成的x域图像的判别结果
    dy_real = discriminator(image=y_img, reuse=True, name='discriminator_y')  # 判别器返回的对真实的y域图像的判别结果
    dx_real = discriminator(image=x_img, reuse=True, name='discriminator_x')  # 判别器返回的对真实的x域图像的判别结果

    gen_loss = gan_loss(dy_fake, tf.ones_like(dy_fake)) + gan_loss(dx_fake,
                                                                   tf.ones_like(dx_fake)) + args.lamda * l1_loss(x_img,
                                                                                                                 fake_x_) + args.lamda * l1_loss(
        y_img, fake_y_)  # 计算生成器的loss

    dy_loss_real = gan_loss(dy_real, tf.ones_like(dy_real))  # 计算判别器判别的真实的y域图像的loss
    dy_loss_fake = gan_loss(dy_fake, tf.zeros_like(dy_fake))  # 计算判别器判别的生成的y域图像的loss
    dy_loss = (dy_loss_real + dy_loss_fake) / 2  # 计算判别器判别的y域图像的loss

    dx_loss_real = gan_loss(dx_real, tf.ones_like(dx_real))  # 计算判别器判别的真实的x域图像的loss
    dx_loss_fake = gan_loss(dx_fake, tf.zeros_like(dx_fake))  # 计算判别器判别的生成的x域图像的loss
    dx_loss = (dx_loss_real + dx_loss_fake) / 2  # 计算判别器判别的x域图像的loss

    dis_loss = dy_loss + dx_loss  # 计算判别器的loss

    gen_loss_sum = tf.summary.scalar("final_objective", gen_loss)  # 记录生成器loss的日志

    dx_loss_sum = tf.summary.scalar("dx_loss", dx_loss)  # 记录判别器判别的x域图像的loss的日志
    dy_loss_sum = tf.summary.scalar("dy_loss", dy_loss)  # 记录判别器判别的y域图像的loss的日志
    dis_loss_sum = tf.summary.scalar("dis_loss", dis_loss)  # 记录判别器的loss的日志
    discriminator_sum = tf.summary.merge([dx_loss_sum, dy_loss_sum, dis_loss_sum])

    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())  # 日志记录器

    g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]  # 所有生成器的可训练参数
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]  # 所有判别器的可训练参数

    lr = tf.placeholder(tf.float32, None, name='learning_rate')  # 训练中的学习率
    d_optim = tf.train.AdamOptimizer(lr, beta1=args.beta1)  # 判别器训练器
    g_optim = tf.train.AdamOptimizer(lr, beta1=args.beta1)  # 生成器训练器

    d_grads_and_vars = d_optim.compute_gradients(dis_loss, var_list=d_vars)  # 计算判别器参数梯度
    d_train = d_optim.apply_gradients(d_grads_and_vars)  # 更新判别器参数
    g_grads_and_vars = g_optim.compute_gradients(gen_loss, var_list=g_vars)  # 计算生成器参数梯度
    g_train = g_optim.apply_gradients(g_grads_and_vars)  # 更新生成器参数

    train_op = tf.group(d_train, g_train)  # train_op表示了参数更新操作
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 设定显存不超量使用
    sess = tf.Session(config=config)  # 新建会话层
    init = tf.global_variables_initializer()  # 参数初始化器

    sess.run(init)  # 初始化所有可训练参数

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=50)  # 模型保存器

    counter = 0  # counter记录训练步数

    for epoch in range(args.epoch):  # 训练epoch数
        shuffle(x_datalists)  # 每训练一个epoch，就打乱一下x域图像顺序
        shuffle(y_datalists)  # 每训练一个epoch，就打乱一下y域图像顺序
        lrate = args.base_lr if epoch < args.epoch_step else args.base_lr * (args.epoch - epoch) / (
        args.epoch - args.epoch_step)  # 得到该训练epoch的学习率
        for step in range(len(x_datalists)):  # 每个训练epoch中的训练step数
            counter += 1
            x_image_resize, y_image_resize = TrainImageReader(x_datalists, y_datalists, step,
                                                              args.image_size)  # 读取x域图像和y域图像
            batch_x_image = np.expand_dims(np.array(x_image_resize).astype(np.float32), axis=0)  # 填充维度
            batch_y_image = np.expand_dims(np.array(y_image_resize).astype(np.float32), axis=0)  # 填充维度
            feed_dict = {lr: lrate, x_img: batch_x_image, y_img: batch_y_image}  # 得到feed_dict
            gen_loss_value, dis_loss_value, _ = sess.run([gen_loss, dis_loss, train_op],
                                                         feed_dict=feed_dict)  # 得到每个step中的生成器和判别器loss
            if counter % args.save_pred_every == 0:  # 每过save_pred_every次保存模型
                save(saver, sess, args.snapshot_dir, counter)
            if counter % args.summary_pred_every == 0:  # 每过summary_pred_every次保存训练日志
                gen_loss_sum_value, discriminator_sum_value = sess.run([gen_loss_sum, discriminator_sum],
                                                                       feed_dict=feed_dict)
                summary_writer.add_summary(gen_loss_sum_value, counter)
                summary_writer.add_summary(discriminator_sum_value, counter)
            if counter % args.write_pred_every == 0:  # 每过write_pred_every次写一下训练的可视化结果
                fake_y_value, fake_x__value, fake_x_value, fake_y__value = sess.run([fake_y, fake_x_, fake_x, fake_y_],
                                                                                    feed_dict=feed_dict)  # run出网络输出
                write_image = get_write_picture(x_image_resize, y_image_resize, fake_y_value, fake_x__value,
                                                fake_x_value, fake_y__value)  # 得到训练的可视化结果
                write_image_name = args.out_dir + "/out" + str(counter) + ".png"  # 待保存的训练可视化结果路径与名称
                cv2.imwrite(write_image_name, write_image)  # 保存训练的可视化结果
            print('epoch {:d} step {:d} \t gen_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, gen_loss_value,
                                                                                        dis_loss_value))


if __name__ == '__main__':
    main()  