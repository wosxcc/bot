import numpy as np
import cv2
import time
import datetime
import math

cap = cv2.VideoCapture('E:/BOT_Car/bot_car/test2.mp4')

kernelss=np.ones((2,2),np.uint8)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2()     # 创建背景减法器
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame1 = np.zeros((640, 480))
# out = cv2.VideoWriter(datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.avi', fourcc, 5.0,
#                       np.shape(frame1))



track_lists=[]
# start_bbox         起始位置
# bbox 框
# id;               // 目标编号
# apperCount;       // 累计出现次数
# type_flag;        // 目标类型标识：1，行人；2，非机动车；3，普通车辆
# last_time;					//最后一次出现时间
# total_time;					//第一次出现到现在时间
# bangle = 180.0;				//方向
# occupy = 0;					//是否被使用
# balert = 0;					//是否需要警报




while (1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    # fgmask = cv2.erode(fgmask, kernelss, iterations=1)
    # fgmask = cv2.erode(fgmask, kernelss, iterations=1)
    ret, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
    fgmask = cv2.medianBlur(fgmask, 5)
    # fgmask = cv2.dilate(fgmask, kernelss, iterations=1)
    # fgmask = cv2.dilate(fgmask, kernelss, iterations=1)
    fgmask =cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernelss)
    ret, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
    cv2.imshow('fgmask',fgmask)
    (_, cnts, _) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #寻找图像轮廓
    maxArea = 0
    xtime =datetime.datetime.now()
    # print('里面存放什么东西',cnts)
    # print(cnts[0 ].shape)
    max_id=0
    for list in track_lists:
        max_id = max(max_id, list['id'])
        if list['total_time']-list['last_time']>20:
            print('删除了ID',list['id'])
            track_lists.remove(list)
            continue
        list['total_time'] +=1
        list['occupy'] = 0

    max_id +=1

    for c in cnts:
        Area = cv2.contourArea(c) #计算轮廓面积
        if Area < maxArea:
            # if cv2.contourArea(c) < 500:
            (x, y, w, h) = (0, 0, 0, 0)
            continue
        else:
            if Area < 300:
                (x, y, w, h) = (0, 0, 0, 0)
                continue
            else:
                maxArea = Area
                m = c   # m为存放的一组坐标
                # print('看看m是什么',m)
                (x, y, w, h) = cv2.boundingRect(m) #获取最小外接矩形边框


        now_id = 0
        if len(track_lists) == 0:
            atrack = {}
            print('空列表中添加了ID', max_id)
            atrack['id']=max_id
            atrack['start_bbox'] = [x, y, w, h]
            atrack['bbox'] = [x, y, w, h]
            atrack['apperCount'] = 1
            atrack['type_flag'] = 1
            atrack['total_time'] = 1
            atrack['last_time'] = atrack['total_time']
            atrack['bangle'] = 180
            atrack['occupy'] = 1
            atrack['balert'] = 0
            track_lists.append(atrack)
            max_id += 1
        else:
            min_range = 10
            index =0
            for list in track_lists:
                if list['occupy'] == 1:
                    print('跳过了ID',list['id'])
                    continue
                y_line = float(abs(y-list['bbox'][1]))/float(fgmask.shape[0])
                x_line = float(abs(x-list['bbox'][0]))/float(fgmask.shape[1])
                line = math.sqrt(y_line*y_line+x_line*x_line)
                weight_b =float(list['bbox'][2]+w)/2.0/float(fgmask.shape[1])
                if line/weight_b<min_range:
                    print(list['id'])
                    min_range = line/weight_b
                    now_id =index
                index += 1

            if min_range!=10 and min_range <0.8:       #1*float(track_lists[now_id]['total_time']-track_lists[now_id]['last_time'])/2
                print('更新列表中ID', track_lists[now_id]['id'])
                track_lists[now_id]['bbox'] = [x, y, w, h]
                track_lists[now_id]['apperCount'] += 1
                track_lists[now_id]['last_time'] = track_lists[now_id]['total_time']
                atrack['occupy'] = 1

            else:
                atrack={}
                print('在列表中添加了ID', max_id)
                atrack['id'] = max_id
                atrack['bbox'] = [x, y, w, h]
                atrack['start_bbox'] = [x, y, w, h]
                atrack['apperCount'] = 1
                atrack['type_flag'] = 1
                atrack['total_time'] = 1
                atrack['last_time'] = atrack['total_time']
                atrack['bangle'] = 180
                atrack['occupy'] = 1
                atrack['balert'] = 0
                now_id = len(track_lists)
                track_lists.append(atrack)
                max_id += 1
        print(track_lists)
        if track_lists[now_id]['apperCount']>3:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,'ID:{0}|--|{1}'.format(str(track_lists[now_id]['id']),str(track_lists[now_id]['apperCount'])), (x, y+20),  cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2)
        # out.write(frame)
    print('耗时:', datetime.datetime.now() - xtime)
    cv2.imshow('frame', frame)
    # cv2.waitKey()
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# out.release()
cap.release()
cv2.destoryAllWindows()