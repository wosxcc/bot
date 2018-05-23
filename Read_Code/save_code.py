# -*-coding:utf-8-*-
from imp import reload
import sys
import  os
import time
import cv2 as cv
import math
import json
import random
from PIL import Image

# split verify_code image
save_dir = "./xcc/"
path= 'H:/Chrome_drown/caffe_verify_code/train_notsplit'
for file in os.listdir(path):
    img = Image.open(path + '/' + file)
    # img = cv.imread(path + '/' + file, 0)
    s = path + '/' + file
    if file.rfind('/') != -1:
        s = file[file.rfind('/') + 1:]
    part = s.split('-')
    id = part[0]
    # img.size (85,26)
    region = [(0, 0, 20, 26), (16, 0, 38, 26), (34, 0, 56, 26), (52, 0, 74, 26)]
    # print(img.crop(region[0]))
    # cv.imshow(region[0])
    # cv.waitKey()
    for i in range(0, len(region)):
        cropImg = img.crop(region[i])
        path = save_dir + str(i) + '-' + part[1][i:i + 1] + '.jpg'
        print(path)
        cropImg.save(path)
