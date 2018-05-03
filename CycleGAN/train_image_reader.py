import os

import numpy as np
import tensorflow as tf
import cv2


def TrainImageReader(x_file_list, y_file_list, step, size):  # 训练数据读取接口
    file_length = len(x_file_list)  # 获取图片列表总长度
    line_idx = step % file_length  # 获取一张待读取图片的下标
    x_line_content = x_file_list[line_idx]  # 获取一张x域图片路径与名称
    y_line_content = y_file_list[line_idx]  # 获取一张y域图片路径与名称
    x_image = cv2.imread(x_line_content, 1)  # 读取一张x域的图片
    y_image = cv2.imread(y_line_content, 1)  # 读取一张y域的图片
    x_image_resize_t = cv2.resize(x_image, (size, size))  # 改变读取的x域图片的大小
    x_image_resize = x_image_resize_t / 127.5 - 1.  # 归一化x域的图片
    y_image_resize_t = cv2.resize(y_image, (size, size))  # 改变读取的y域图片的大小
    y_image_resize = y_image_resize_t / 127.5 - 1.  # 归一化y域的图片
    return x_image_resize, y_image_resize  # 返回读取并处理的一张x域图片和y域图片