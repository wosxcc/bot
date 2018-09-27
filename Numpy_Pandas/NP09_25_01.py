import numpy as np
import random
import cv2 as cv

# def read_img(txt_name):
#     label_lines = []
#     image_lines = []
#     txt_open = open(txt_name)
#     txt_read = txt_open.read()
#     txt_lines = txt_read.split('\n')
#     count =0
#     for line in txt_lines:
#         xlabel = []
#         if len(line)>3:
#             line_list = line.split(' ')
#             img = cv.imread(line_list[0])
#             image_lines.append(img)
#             a_label = [float(i) for i in line_list[1:]]
#             a_label[0] = a_label[0]/2
#             a_label = np.array(a_label,dtype='float32')
#             a_label = (a_label-0.5)*2.0
#             # print(a_label)
#             label_lines.append(a_label)
#         if count>10:
#             break
#         count+=1
#     label_linesc=[[float(i) for i in xline] for xline in label_lines]
#     ximage_lines=np.array(image_lines)
#     xlabel_linesc=np.array(label_linesc, dtype='float32')
#     return ximage_lines,xlabel_linesc
#
#
#
#
# image,label =read_img('E:/xbot/face_into/face_key_point/trainc.txt')
#
# print('image',image)
# print('label',label)
# # random.shuffle()


for ai in range(32):
    print(random.randint(0, 10064 - 1))

