# -*- coding: utf-8 -*-
                    ####抠图软件####
import cv2 as cv
import numpy as np
import  math
import  os


drawing=False
ix,iy=-1,-1             # 初始化位置
rex,rey=-1,-1           # 初始化位置

nclass =['显示器', '键盘', '鼠标', '笔记本电脑', '电话', '打印机', '照相机', '文件架', '笔筒', '水杯', '计算器', '盆栽', '12','13','14','15','16','17','18','19','20','21','22','23','24','25','26']
IMG_W = 800
IMG_H = 800
now_class = 0
windows_name='E:/Desk_Set/55.mp4'   # 文件位置E:\Desk_Set
path='E:/Desk_Set/train1120/'                                     # 文件保存位置
img_copy=[]
sum_init=[]

def draw_circle(event,x,y,flags,param):

    global ix,iy,rex,rey,drawing, now_class
    if event==cv.EVENT_LBUTTONDOWN:
        drawing =True
        rex, rey=x,y
        ix,iy=x,y
    elif event==cv.EVENT_MOUSEMOVE and  flags==cv.EVENT_FLAG_LBUTTON:
        if drawing ==True:
            im_draw = np.copy(img_copy)
            # cv.putText(im_draw, str(now_class), (rex, rey+20),  cv.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2)

            color_b = 127*(now_class//9)
            color_g = 127*((now_class%9)//3)
            color_r = 127*(now_class%3)

            cv.line(im_draw, (rex, rey), (x, y), (0, 255, 255), 2)
            cv.rectangle(im_draw, (rex, rey), (x, y), (color_b, color_g, color_r), 2)
            ix, iy = x, y
            cv.imshow(windows_name, im_draw)
            # else:
            #     im_draw = np.copy(img_copy)
            #     # cv.putText(im_draw,class1,(rex, rey+20),cv.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),2)
            #     cv.line(im_draw, (rex, rey), (x, y), (0, 255, 255), 2)
            #     cv.rectangle(im_draw, (rex, rey), (x, y), (255, 0, 255), 2)
            #     ix, iy = x, y
            #     cv.imshow(windows_name, im_draw)

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
        if xwidth*yheight>=200:
            color_b = 127 * (now_class // 9)
            color_g = 127 * ((now_class % 9) // 3)
            color_r = 127 * (now_class % 3)
            print(nclass[now_class], x, y,'宽', xwidth,'高', yheight)
            sum_init.append([now_class,mx,my,xwidth,yheight])
            # cv.putText(img_copy, class0, (int(mx-xwidth/2), int(my-yheight/2)+20), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0),2)
            cv.rectangle(img_copy, (int(mx-xwidth/2), int(my-yheight/2)), (int(mx+xwidth/2), int(my+yheight/2)), (color_b, color_g, color_r), 1)
            drawing=False
count_c=0

cap=cv.VideoCapture(windows_name)
keep_num=int(len(os.listdir(path[0:-1]))/2)+1120
# keep_num=
while(1):
    ret, img = cap.read()
    now_class = 0
    img=cv.resize(img,(1200,800), interpolation=cv.INTER_CUBIC)
    # imga = cv.imread(windows_name+'/'+file[:-6]+'a.jpeg')
    # cv.imshow('image_a',imga)
    img_copy = np.copy(img)
    count_c+=1
    # if count_c%3==0:
    cv.namedWindow(windows_name)
    cv.setMouseCallback(windows_name,draw_circle)
    sum_init = []

    while(1):
        cv.imshow(windows_name, img_copy)
        k=cv.waitKey(0)&0xFF
        if k>47 and k<58: # c转换类型
            now_class= k-48
            cv.imshow(windows_name, img_copy)
        elif k>64 and k<71:
            now_class =10+ k - 65
            cv.imshow(windows_name, img_copy)
        elif k==ord('s'):       # s保存
            if len(sum_init)>0:
                creat_name=str(keep_num)
                for i in range(int(5-len(creat_name))):
                    creat_name='0'+creat_name
                imgs = cv.resize(img, (IMG_W, IMG_H), interpolation=cv.INTER_CUBIC)
                cv.imwrite(path+creat_name+'.jpg',imgs)
                write_txt = open(path+creat_name + '.txt', 'w')
                output_txt=''
                for bbox in sum_init:
                    output_txt+=str(bbox[0])+' '+str(bbox[1]/img.shape[1])+' '+str(bbox[2]/img.shape[0])+' '+str(bbox[3]/img.shape[1])+' '+str(bbox[4]/img.shape[0])+'\n'
                write_txt.write(output_txt)
                write_txt.close()

                # os.remove(windows_name + '/' + file)
                print('保存成功：',sum_init)
                keep_num+=1
                break
        elif k ==ord('r'):
            if len(sum_init)>0:
                sum_init=np.delete(sum_init, -1, axis=0)    # axis=0，表示行，axis=1，表示列
                sum_init = sum_init.tolist()                # 列标转为数组要转回去不然会报错
                img_copy =np.copy(img)
                for sline  in sum_init:
                    color_b = 127 * (sline[0] // 9)
                    color_g = 127 * ((sline[0] % 9) // 3)
                    color_r = 127 * (sline[0] % 3)
                    cv.rectangle(img_copy, (int(sline[1] - sline[3]  / 2), int(sline[2]  - sline[4]  / 2)),
                                 (int(sline[1] + sline[3] / 2), int(sline[2] + sline[4] / 2)), (color_b, color_g, color_r), 1)
        elif k == ord('t'):             # 清屏
            img_copy = np.copy(img)
            sum_init=[]
        elif k==32:  # 空格跳到下一帧
            break
cv.destroyAllWindows()
