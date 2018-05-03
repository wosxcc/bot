import cv2 as cv
import numpy as np
img= cv.imread('E:/Face_Hand/train/02973.jpg')

sss='''0 0.133 0.47733333333333333 0.014 0.037333333333333336
0 0.219 0.472 0.022 0.037333333333333336
0 0.783 0.5053333333333333 0.038 0.056'''
bboxs=sss.split('\n')
print(bboxs)
for box in bboxs:
    xbox=box.split(' ')
    xbox=[float(i) for i in xbox]
    classname='posen'
    if xbox[0]==1:
        classname='car'

    # cv.putText(img,classname , (int(xbox[1] - xbox[3] / 2), int(xbox[2] - xbox[4] / 2) + 20), cv.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
    # img=cv.rectangle(img, (int(xbox[1] - xbox[3] / 2), int(xbox[2] - xbox[4] / 2)), (int(xbox[1] + xbox[3] / 2), int(xbox[2] + xbox[4] / 2)),(255, 0, 0), 2)

    # cv.putText(img, classname, (int((xbox[1] - xbox[3])*img.shape[1]), int((xbox[2] - xbox[4])*img.shape[0]) + 20), cv.FONT_HERSHEY_SIMPLEX,
    #            1, (255, 0, 0), 2)
    img = cv.rectangle(img, (int((xbox[1] - xbox[3]/2)*img.shape[1]), int((xbox[2] - xbox[4]/2)*img.shape[0])),
                       (int((xbox[1] + xbox[3]/2)*img.shape[1]), int((xbox[2] + xbox[4]/2)*img.shape[0])), (255, 0, 0), 2)
    # print('xbox',xbox)
    # print((int(xbox[1] - xbox[3] / 2), int(xbox[1] + xbox[3] / 2)), int(xbox[2] - xbox[4] / 2), int(xbox[2] + xbox[4] / 2))
cv.imshow('img',img)
cv.waitKey()