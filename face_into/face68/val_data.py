import os
import cv2 as cv

txt_file='E:/face68/train.txt'
txt_open=open(txt_file)
txt_read=txt_open.read()
txt_line= txt_read.split('\n')

for aimg in txt_line:

    if len(aimg)>3:
        img_line=aimg.split(' ')
        print(len(img_line))
        img=cv.imread('E:/face68/trainb/'+img_line[0])
        img = cv.resize(img, (480, 480), interpolation=cv.INTER_CUBIC)
        spot_line =[float(i) for i in img_line[1:]]
        for  i in  range(int(len(spot_line)/8)):
            cv.circle(img,(int(spot_line[8*i]*img.shape[1]),int(spot_line[8*i+1]*img.shape[0])),2,(0,0,255),-1)


        cv.imshow(img_line[0],img)
        cv.waitKey()
        cv.destroyAllWindows()

