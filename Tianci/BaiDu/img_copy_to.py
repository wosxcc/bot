import numpy as np
import os ,shutil
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mping
import cv2 as cv
from scipy import misc
txt_read =open('train3.txt').read()
class1 =[[70,130,180],[220,20,60],[128,0,128],[255,0,0],[0,0,60],[0,60,100]]
class2 =[[0,0,142] ,[119,11,32],[244,35,232],[0,0,160]]
class3 =[[153,153,153],[220,220,0],[250,170,30]]
class4 =[[102,102,156],[128,0,0]]
class5 =[[128,64,128],[238,232,170]]
class6 =[[190,153,153]]
class7 =[[0,0,230],[128,128,0],[128,78,160],[150,100,100],[255,165,0],[180,165,180],[107,142,35],[201,255,229],[0,191,255],[51,255,51],[250,128,114],[127,255,0]]
class8 =[[255, 128, 0] ,[0, 255, 255] ,[178, 132,190] ,[128, 128,64] ,[102, 0,204]]
class0 =[[0,153,153],[255,255,255]]


cla0=[0,249,255]
cla1=[200,204,213,209,206,207]
cla2=[201,203,211,208]
cla3=[216,217,215]
cla4=[218,219]
cla5=[210,232]
cla6=[214]
cla7=[202,220,221,222,231,224,225,226,230,228,229,233]
cla8=[205,212,227,223,250]





countss =30000;
for img_group  in txt_read.split('\n'):
    img1 =cv.resize(cv.imread(img_group.split('---')[0]),(846,428),cv.INTER_CUBIC)
    img2 = cv.resize(cv.imread(img_group.split('---')[1]),(846,428),cv.INTER_CUBIC)
    img_p = Image.open(img_group.split('---')[1])
    img_p=img_p.resize((846,428))
    # print('哈哈',img_p.getpixel((0, 0)))
    #img_p = misc.imresize(img_p,(846,428))
    # print(img_p.size)
    img_gray = np.zeros([428, 846], np.uint8)
    for y in range(428):
        for x in range(846):
            # if img_p.getpixel((x,y))>0:
            #     print(img_p.getpixel((x,y)),'img_p.getpixel((x,y)) in cla0',img_p.getpixel((x,y)) in cla0)
            if img_p.getpixel((x,y)) in cla1:
                img_gray[y, x] =1
            if img_p.getpixel((x, y)) in cla2:
                img_gray[y, x] = 2
            if img_p.getpixel((x, y)) in cla3:
                img_gray[y, x] = 3
            if img_p.getpixel((x, y)) in cla4:
                img_gray[y, x] = 4
            if img_p.getpixel((x, y)) in cla5:
                img_gray[y, x] = 5
            if img_p.getpixel((x, y)) in cla6:
                img_gray[y, x] = 6
            if img_p.getpixel((x, y)) in cla7:
                img_gray[y, x] = 7
            if img_p.getpixel((x, y)) in cla8:
                img_gray[y, x] = 8
    #         print(img2[y, x].tolist())
    #         print(img2[y,x].tolist() in class1)
    #         if img2[y,x].tolist() in class1:
    #             img_gray[y, x] = 255
    #         if img2[y, x].tolist() in class2:
    #             img_gray[y, x] = 255
    #         if img2[y, x].tolist() in class3:
    #             img_gray[y, x] = 255
    #         if img2[y, x].tolist() in class4:
    #             img_gray[y, x] = 255
    #         if img2[y, x].tolist() in class5:
    #             img_gray[y, x] = 255
    #         if img2[y, x].tolist() in class6:
    #             img_gray[y, x] = 255
    #         if img2[y, x].tolist() in class7:
    #             img_gray[y, x] = 255
    #         if img2[y, x].tolist() in class8:
    #             img_gray[y, x] = 255

            # if img2[y,x,:]==[70,130,180] or img2[y,x,:]==[220,20,60] or img2[y,x,:]==[128,0,128] or img2[y,x,:]==[255,0,0] or img2[y,x,:]==[0,0,60] or img2[y,x,:]==[0,60,100]:
            #     img_gray[y,x]=30
            # if img2[y,x,:]==[0,0,142] or img2[y,x,:]==[119,11,32] or img2[y,x,:]==[244,35,232] or img2[y,x,:]==[0,0,160]:
            #     img_gray[y,x]=60
            # if img2[y,x,:]==[153,153,153] or img2[y,x,:]==[220,220,0] or img2[y,x,:]==[250,170,30]:
            #     img_gray[y,x]=90
            # if img2[y,x,:]==[102,102,156] or img2[y,x,:]==[128,0,0]:
            #     img_gray[y,x]=120
            # if img2[y,x,:]==[128,64,128]or img2[y,x,:]==[238,232,170]:
            #     img_gray[y,x]=150
            # if img2[y,x,:]==[190,153,153]:
            #     img_gray[y,x]=180
            # if img2[y,x,:]==[0,0,230] or img2[y,x,:]==[128,128,0] or img2[y,x,:]==[128,78,160] or img2[y,x,:]==[150,100,100] or img2[y,x,:]==[255,165,0] or img2[y,x,:]==[180,165,180]:
            #     img_gray[y,x]=210
            # if img2[y,x,:]==[107,142,35] or img2[y,x,:]==[201,255,229] or img2[y,x,:]==[0,191,255] or img2[y,x,:]==[51,255,51] or img2[y,x,:]==[250,128,114] or img2[y,x,:]==[127,255,0]:
            #     img_gray[y,x]=210
            # if img2[y, x, :] == [255, 128, 0] or img2[y, x, :] == [0, 255, 255] or img2[y, x, :] == [178, 132,190] or img2[y, x,] == [128, 128,64] or img2[y, x,] == [102, 0,204]:
            #     img_gray[y, x] = 240


    cname=str(countss)
    for i in range(5-len(cname)):
        cname='0'+cname
    # print(img_gray.tolist())

    cv.imwrite('E:/Model/deeplab/Database/JPEGImages/'+cname+'.jpg',img1)
    cv.imwrite('E:/Model/deeplab/Database/SegmentationClass/' + cname + '.png', img_gray)
    # cv.imshow('img1', img1)
    # cv.imshow('img2', img2)
    # cv.imshow('img_gray', img_gray)
    countss+=1
    cv.waitKey()
    cv.destroyAllWindows()

