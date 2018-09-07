# -*- coding: utf-8 -*-
                    ####抠图软件####
import cv2 as cv
import numpy as np
import  math
import  os
import sys

drawing=False
mode=True
ix,iy=-1,-1             # 初始化位置
rex,rey=-1,-1           # 初始化位置

class0='hand'
class1='face'

IMG_W = 480
IMG_H = 480

windows_name='E:/BOT_Person/bot_face/222.mp4'   # 视频文件位置
cap=cv.VideoCapture(windows_name)
path='./train/'                                     # 文件保存位置
img_copy=[]
sum_init=[]

def draw_circle(event,x,y,flags,param):

    global ix,iy,rex,rey,drawing,mode
    if event==cv.EVENT_LBUTTONDOWN:
        drawing =True
        rex, rey=x,y
        ix,iy=x,y
    elif event==cv.EVENT_MOUSEMOVE and  flags==cv.EVENT_FLAG_LBUTTON:
        if drawing ==True:
            if mode==True:
                im_draw = np.copy(img_copy)
                # cv.putText(im_draw, class0, (rex, rey+20),  cv.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2)
                cv.line(im_draw, (rex, rey), (x, y), (0, 255, 255), 2)
                cv.rectangle(im_draw, (rex, rey), (x, y), (255, 0, 0), 2)
                ix, iy = x, y
                cv.imshow(windows_name, im_draw)
            else:
                im_draw = np.copy(img_copy)
                # cv.putText(im_draw,class1,(rex, rey+20),cv.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),2)
                cv.line(im_draw, (rex, rey), (x, y), (0, 255, 255), 2)
                cv.rectangle(im_draw, (rex, rey), (x, y), (255, 0, 255), 2)
                ix, iy = x, y
                cv.imshow(windows_name, im_draw)

count_c=0
keep_num=int(len(os.listdir(path[0:-1]))/2)+85000

while (1):
    cv.imshow(windows_name, img_copy)
# keep_num=
while(1):
    ret, img = cap.read()
    count_c += 1
    print(count_c)
    if count_c > 0: #21250
        img=cv.resize(img,(1000,800),cv.INTER_CUBIC)
        img_copy = np.copy(img)

        # if count_c%3==0:
        cv.namedWindow(windows_name)
        cv.setMouseCallback(windows_name,draw_circle)
        sum_init = []
        mode = True
        while(1):
            cv.imshow(windows_name, img_copy)

            k=cv.waitKey(0)&0xFF

            if k==ord('c'): # c转换类型
                print('转换成功')
                mode=not mode
                cv.imshow(windows_name, img_copy)
            elif k==ord('s'):       # s保存
                if len(sum_init)>0:
                    creat_name=str(keep_num)
                    for i in range(int(5-len(creat_name))):
                        creat_name='0'+creat_name
                    imgs =cv.resize(img, (IMG_W,IMG_H), interpolation=cv.INTER_CUBIC)
                    cv.imwrite(path+creat_name+'.jpg',imgs)
                    write_txt = open(path+creat_name+'.txt', 'w')
                    output_txt=''
                    for bbox in sum_init:
                        output_txt+=str(bbox[0])+' '+str(bbox[1]/img.shape[1])+' '+str(bbox[2]/img.shape[0])+' '+str(bbox[3]/img.shape[1])+' '+str(bbox[4]/img.shape[0])+'\n'
                    write_txt.write(output_txt)
                    write_txt.close()
                    print('保存成功：',sum_init)
                    keep_num+=1
                    break
            elif k ==ord('r'):
                if len(sum_init)>0:
                    sum_init=np.delete(sum_init, -1, axis=0)    # axis=0，表示行，axis=1，表示列
                    sum_init = sum_init.tolist()                # 列标转为数组要转回去不然会报错
                    img_copy =np.copy(img)
                    for sline  in sum_init:
                        cv.rectangle(img_copy, (int(sline[1] - sline[3]  / 2), int(sline[2]  - sline[4]  / 2)),
                                     (int(sline[1] + sline[3] / 2), int(sline[2] + sline[4] / 2)), (255, 0, 0), 1)
            elif k == ord('d'):             # 清屏
                img_copy = np.copy(img)
                sum_init = []
            elif k == 27:
                sys.exit()
            elif k==32:  # 空格跳到下一帧
                break
cv.destroyAllWindows()
