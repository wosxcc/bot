import os
import cv2 as cv


train_open=open('trainb.txt')
train_read=train_open.read()
print(len(train_read.split('\n')))
for line in train_read.split('\n'):
    if len(line)>3:
        txt_init=line.split(' ')
        img= cv.imread(txt_init[0])
        txt_float = [float(i) for i in txt_init[1:]]
        biaoq = 'xiao'
        if txt_float[0] == 0:
            biaoq = 'buxiao'
        elif txt_float[0] == 2:
            biaoq = 'daxiao'
        biaoq += str(txt_float[1])
        img = cv.putText(img, biaoq, (0, 25), 2, cv.FONT_HERSHEY_PLAIN, (255, 0, 0))
        for x in range(14):
            img = cv.circle(img, (int(txt_float[116+2 + x * 2] * img.shape[1]), int(txt_float[ 116+2 + x * 2 + 1] * img.shape[0])),
                            1, (0, 255, 0), -1)
        cv.imshow('img',img)
        cv.waitKey()
