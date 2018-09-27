# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from PIL import ImageGrab, Image, ImageDraw, ImageFont

# 图片上写中文函数（图片，文字，  起始下，   起始有，    文字大小,    颜色）
def dram_chinese(img,chinese,char_x=20,char_y=20,char_size=20 ,fillColor = (255, 255, 0)): #   返回写完文字后的图片
    img_PIL = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    font = ImageFont.truetype("simhei.ttf", char_size, encoding="utf-8")
    # fillColor = (255, 255, 0)
    position = (char_x, char_y)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=fillColor)
    img = cv.cvtColor(np.asarray(img_PIL), cv.COLOR_RGB2BGR)
    return img




# 图像旋转函数 （图片 ，旋转中心 ，   旋转角度  ，缩放）
def img_rotate(img, center=None, angle =0, scale=1.0):  # 返回旋转后图片
    img_w = img.shape[1]
    img_h = img.shape[0]
    if center==None:
        center =(int(img_w/2),int(img_h/2))
    M = cv.getRotationMatrix2D(center, angle, scale)
    imgs = cv.warpAffine(img, M, (img_w, img_h))
    return imgs


# 图像卷积   （图片 ，卷积核，步长， 填充）
def img_conv(img, kernel,stride,padding = 0):

    kernel = np.array(kernel, dtype='float32')
    kernel_size = kernel.shape[0]
    if padding ==1:
        padd_number= int((kernel-1)/2)

        if len(img.shape)==2:
            imgb = np.zeros((img.shape[0]+2*padd_number,img.shape[1]+2*padd_number),np.float32)
            imgb[padd_number:-padd_number,padd_number:-padd_number] =img
        else:
            imgb = np.zeros((img.shape[0] + 2 * padd_number, img.shape[1] + 2 * padd_number, img.shape[2]),np.float32)
            imgb[padd_number:-padd_number, padd_number:-padd_number,:] = img
    else:
        imgb =img

    img_w = int((img.shape[1] - kernel_size + 1) // stride)
    img_h = int((img.shape[0] - kernel_size + 1) // stride)
    if len(img.shape) == 2:
        imgr =np.zeros((img_h,img_w),np.uint8)
        imconv = np.zeros((img_h,img_w),np.float32)
        for x in range(img_w):
            for y in range(img_h):
                imconv[y, x] = np.sum(
                    np.dot(imgb[y * stride:y * stride + kernel_size, x * stride:x * stride + kernel_size], kernel))
    else:
        imgr = np.zeros((img_h, img_w,img.shape[2]), np.uint8)
        imgb = imgb[:, :, :].transpose((2, 0, 1))
        imconv = np.zeros((img.shape[2],img_h, img_w), np.float32)
        for z in range(img.shape[2]):
            for x in range(img_w):
                for y in range(img_h):
                    imconv[z, y, x] = np.sum(
                        np.dot(imgb[z,y * stride:y * stride + kernel_size, x * stride:x * stride + kernel_size], kernel))
        imconv =imconv.transpose((1, 2, 0))
    imconv = imconv - np.min(imconv)
    imconv = imconv / np.max(imconv) * 255
    imconv = np.array(imconv, dtype='int')
    if len(img.shape) == 2:
        imgr[:,:] = imconv
    else:
        imgr[:, :,:] = imconv
    print('卷积结束')
    return imgr










