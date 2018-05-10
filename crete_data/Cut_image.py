# -*- coding: utf-8 -*-
                    ####抠图软件####
import cv2 as cv
import numpy as np
import  math
import  os


drawing=False
mode=True
ix,iy=-1,-1             # 初始化位置
rex,rey=-1,-1           # 初始化位置

class0='hand'
class1='face'


windows_name='E:/xbot/pachong/person'   # 文件位置
path='./train_person/'                                     # 文件保存位置
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
                cv.putText(im_draw, class0, (rex, rey+20),  cv.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2)
                cv.line(im_draw, (rex, rey), (x, y), (0, 255, 255), 2)
                cv.rectangle(im_draw, (rex, rey), (x, y), (255, 0, 0), 2)
                ix, iy = x, y
                cv.imshow(windows_name, im_draw)
            else:
                im_draw = np.copy(img_copy)
                cv.putText(im_draw,class1,(rex, rey+20),cv.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),2)
                cv.line(im_draw, (rex, rey), (x, y), (0, 255, 255), 2)
                cv.rectangle(im_draw, (rex, rey), (x, y), (255, 0, 255), 2)
                ix, iy = x, y
                cv.imshow(windows_name, im_draw)

    if event==cv.EVENT_LBUTTONUP:
        # img_copy = np.copy(img)
        x = max(x, 0)
        y = max(y, 0)
        x = min(x, img_copy.shape[1])
        y = min(y, img_copy.shape[0])
        mx=(rex + x) / 2.0
        my=(rey + y) / 2.0
        min_x = rex if (x > rex) else x
        max_x = rex if (x < rex) else x
        min_y = rey if (y > rey) else y
        max_y = rey if (y < rey) else y
        xwidth = abs(rex - x)
        yheight = abs(rey - y)
        if xwidth*yheight>=400:
            # print('结束时的位置：rex,rey,x,y',mx,my,xwidth,yheight)
            if mode == True:                                # 如果是人类型是0
                sum_init.append([0,mx,my,xwidth,yheight])
                cv.putText(img_copy, class0, (int(mx-xwidth/2), int(my-yheight/2)+20), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0),2)
                cv.rectangle(img_copy, (int(mx-xwidth/2), int(my-yheight/2)), (int(mx+xwidth/2), int(my+yheight/2)), (255, 0, 0), 2)
            else:                                            # 如果是车类型是1
                sum_init.append([1,mx,my,xwidth,yheight])
                cv.putText(img_copy, class1, (int(mx - xwidth / 2), int(my - yheight / 2)+20), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 0, 255), 2)
                cv.rectangle(img_copy, (int(mx - xwidth / 2), int(my - yheight / 2)),
                             (int(mx + xwidth / 2), int(my + yheight / 2)), (255, 0, 255), 2)
            drawing=False
count_c=0
keep_num=int(len(os.listdir(path[0:-1]))/2)+10000
# keep_num=
for file in os.listdir(windows_name):
    img = cv.imread(windows_name+'/'+file)
    print(file)
    img=cv.resize(img,(800,600), interpolation=cv.INTER_CUBIC)
    img_copy = np.copy(img)
    count_c+=1
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
                cv.imwrite(path+creat_name+'.jpg',img)
                write_txt = open(path+creat_name+'.txt', 'w')
                output_txt=''
                for bbox in sum_init:
                    output_txt+=str(bbox[0])+' '+str(bbox[1]/img.shape[1])+' '+str(bbox[2]/img.shape[0])+' '+str(bbox[3]/img.shape[1])+' '+str(bbox[4]/img.shape[0])+'\n'
                write_txt.write(output_txt)
                write_txt.close()
                print('保存成功：',sum_init)
                keep_num+=1
                break
        elif k == ord('d'):             # 清屏
            img_copy = np.copy(img)
            sum_init=[]
        elif k==32:  # 空格跳到下一帧
            break
cv.destroyAllWindows()
