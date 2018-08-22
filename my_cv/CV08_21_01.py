import numpy as np
import cv2 as cv


step = 50000/10
img_H = 1000
img_W = 1200
coordinate = np.zeros((img_H,img_W,3),np.uint8)
coordinate[:, :, :] = 255
line_c =8
coordinate=cv.line(coordinate,(100,img_H-100),(img_W,img_H-100),(0,0,0),2)
coordinate=cv.line(coordinate,(100,0),(100,img_H-100),(0,0,0),2)

for i in range(11):
    coordinate = cv.line(coordinate, (i * 100 + 100, img_H - 100), (i * 100 + 100, 0), (0, 0, 0), 1)
    coordinate = cv.line(coordinate, (100, i * 100 + 100), (img_W, i * 100 + 100), (0, 0, 0), 1)
    if i>0:
        cv.putText(coordinate, str(i*step), (i * 100 + 100-32, img_H - 100 + 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    biaohao = '%.1f' % (1.0 - i * 0.1 - 0.2)
    if biaohao=='-0.0':
        cv.putText(coordinate, '0', (100 - 50, i * 100 + 100 + 10+30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        cv.putText(coordinate, biaohao, (100 - 50, i * 100 + 100 + 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

cv.imshow('coordinate',coordinate)
cv.waitKey()
cv.destroyAllWindows()
