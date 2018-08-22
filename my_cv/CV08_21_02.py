import numpy as np
import cv2 as cv

def draw_form(MAX_STEP):
    step = MAX_STEP / 10
    img_H = 1000
    img_W = 1200
    coordinate = np.zeros((img_H, img_W, 3), np.uint8)
    coordinate[:, :, :] = 255
    line_c = 8
    coordinate = cv.line(coordinate, (100, img_H - 100), (img_W, img_H - 100), (0, 0, 0), 2)
    coordinate = cv.line(coordinate, (100, 0), (100, img_H - 100), (0, 0, 0), 2)

    for i in range(11):
        coordinate = cv.line(coordinate, (i * 100 + 100, img_H - 100), (i * 100 + 100, 0), (0, 0, 0), 1)
        coordinate = cv.line(coordinate, (100, i * 100 + 100), (img_W, i * 100 + 100), (0, 0, 0), 1)
        if i > 0:
            cv.putText(coordinate, str(i * step), (i * 100 + 100 - 32, img_H - 100 + 50), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 0, 0), 2)
        biaohao = '%.1f' % (1.0 - i * 0.1 - 0.2)
        if biaohao == '-0.0':
            cv.putText(coordinate, '0', (100 - 50, i * 100 + 100 + 10 + 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv.putText(coordinate, biaohao, (100 - 50, i * 100 + 100 + 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return coordinate


def drow_spot(img,x,y,MAX_STEP):
    for i in range(x.shape[0]):
        spot_x = int(x[i]/MAX_STEP*1000+100)
        spot_y =int(900-y[i]*1000)
        print('画点位置：',spot_x,spot_y)
        cv.circle(img,(spot_x,spot_y),3,(0,0,255),-1)
        cv.imshow('LOSS',img)
        # cv.waitKey(0)

MAX_STEP=50000
imgs =draw_form(MAX_STEP)

x = [10000,20000,25000,30000]
y = [0.6832,0.5475413,0.3315,0.254152]
drow_spot(imgs,np.array(x),np.array(y),MAX_STEP)
print('结束了么')
