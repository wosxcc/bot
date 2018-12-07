import cv2 as cv
import os
import numpy as np


jimg_path ='E:/Model/deeplab/Database/JPEGImages'
pimg_path ='E:/Model/deeplab/Database/SegmentationClass/'


for imfile in os.listdir(jimg_path):
    yimg= cv.imread(jimg_path+'/'+imfile)
    print(pimg_path+imfile[:-4]+'.png')
    gimg = cv.imread(pimg_path+imfile[:-4]+'.png',0)




    for y in range(gimg.shape[0]):
        for x in range(gimg.shape[1]):
            if gimg[y,x]!=0:
                yimg[y,x,2]=255
    cv.imshow(jimg_path + '/' + imfile, yimg)
    cv.imshow(pimg_path+imfile[-4]+'.png', gimg)
    cv.waitKey()
    cv.destroyAllWindows()

