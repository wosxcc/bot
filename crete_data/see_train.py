import cv2 as cv
import numpy as np
img= cv.imread('E:/Face_Hand/train/00000.jpg')

sss='''0 0.5915637860082305 0.443 0.03909465020576132 0.046
1 0.3896958459735898 0.25727931715548036 0.048377797191525684 0.050699250668287274'''
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